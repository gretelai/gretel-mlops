"""Script to run Gretel synthetics on source data."""

from kfp.dsl import InputPath, OutputPath, component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-secret-manager",
        "gretel-client[gcp,tuner]",
        "imblearn",
        "optuna",
        "scikit-learn",
        "xgboost",
    ],
)
def gretel_component(
    config: str,
    gretel_secret: str,
    project_number: str,
    input_dir: InputPath(),
    output_dir: OutputPath(),
):
    import json
    import logging
    import os

    import joblib
    import numpy as np
    import optuna
    import pandas as pd 
    import xgboost as xgb
    from google.api_core.exceptions import GoogleAPICallError, PermissionDenied
    from google.cloud import secretmanager
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 mean_absolute_error, mean_squared_error,
                                 precision_recall_curve, r2_score, roc_auc_score)

    from gretel_client import Gretel
    from gretel_client.projects.models import Model
    from gretel_client.tuner import BaseTunerMetric, MetricDirection


    def get_secret(secret_id, project_number):

        """
        Retrieves a secret value from Google Cloud Secret Manager.

        This function initializes a Secret Manager client using the given project number,
        constructs the resource ID of the secret, and then accesses the latest version
        of the specified secret.

        Args:
            secret_id (str): The ID of the secret to retrieve.
            project_number (str): The project number where the secret is stored.

        Returns:
            str: The secret value as a string if the retrieval is successful; None otherwise.
        """
     
        # Initialize the client
        client = secretmanager.SecretManagerServiceClient()
        secret_version = (
            f"projects/{project_number}/secrets/{secret_id}/versions/latest"
        )

        try:
            response = client.access_secret_version(
                request={"name": secret_version}
            )
            gretel_api_key = response.payload.data.decode("UTF-8")
            return gretel_api_key
        except PermissionDenied as e:
            print(f"Permission denied: {e.message}")
            return None
        except GoogleAPICallError as e:
            print(f"An error occurred accessing the secret: {e}")
            return None


    def naive_upsample(df, target_column, target_balance=1.0):
        """
        Upsamples a DataFrame to balance classes in the target column.

        Args:
            df (pd.DataFrame): The DataFrame to upsample.
            target_column (str): The target column for class balancing.
            target_balance (float): The desired balance ratio.

        Returns:
            pd.DataFrame: The upsampled DataFrame.
        """
        # Initialize the over-sampler
        over_sampler = RandomOverSampler(sampling_strategy=target_balance)
        y = df.pop(target_column)
        df_resampled, y_resampled = over_sampler.fit_resample(df, y)
        df_resampled[target_column] = y_resampled

        return df_resampled


    def compute_optimal_f1(y_test, predictions):
        """
        Computes the optimal F1 score for binary classification predictions.

        Args:
            y_test (np.ndarray): True binary labels.
            predictions (np.ndarray): Predicted probabilities.

        Returns:
            tuple: Best F1 score, optimal precision, optimal recall, and
                confusion matrix.
        """
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, predictions)

        # Remove zero precision and recall values
        selection = ~((precision == 0) & (recall == 0))
        precision = precision[selection]
        recall = recall[selection]
        thresholds = thresholds[selection[:-1]]

        # Calculate F1 scores and find the threshold that maximizes it
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1_score = np.max(f1_scores)

        # Calculate precision and recall at the optimal threshold
        optimal_precision = precision[np.argmax(f1_scores)]
        optimal_recall = recall[np.argmax(f1_scores)]

        # Compute confusion matrix at the optimal threshold
        predictions_binary = (predictions >= best_threshold).astype(int)
        conf_matrix = confusion_matrix(y_test, predictions_binary)

        return best_f1_score, optimal_precision, optimal_recall, conf_matrix


    def generate_classification_report(y_test, predictions):
        """
        Generates a report containing various classification metrics.

        Args:
            y_test (np.ndarray): True binary labels.
            predictions (np.ndarray): Predicted probabilities.

        Returns:
            dict: A dictionary containing classification metrics.
        """
        # Compute classification metrics
        f1, precision, recall, conf_matrix = compute_optimal_f1(
            y_test, predictions
        )
        roc_auc = roc_auc_score(y_test, predictions)
        pr_auc = average_precision_score(y_test, predictions)

        # Assemble the metrics into a report dictionary
        report_dict = {
            "metrics": {
                "auc": {"value": roc_auc},
                "aucpr": {"value": pr_auc},
                "precision": {"value": precision},
                "recall": {"value": recall},
                "f1": {"value": f1},
                "confusion_matrix": {"value": conf_matrix.tolist()},
            },
        }

        return report_dict


    def generate_regression_report(y_test, predictions):
        """
        Generates a report containing various regression metrics.

        Args:
            y_test (np.ndarray): True values.
            predictions (np.ndarray): Predicted values.

        Returns:
            dict: A dictionary containing regression metrics.
        """
        # Calculate regression metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Calculate standard deviations for each metric
        mse_std = np.std((y_test - predictions) ** 2)
        rmse_std = np.std(np.abs(y_test - predictions))
        mae_std = np.std(np.abs(y_test - predictions))
        r2_std = np.std(
            1 - ((y_test - predictions) ** 2) / ((y_test - np.mean(y_test)) ** 2)
        )

        # Assemble the metrics into a report dictionary
        report_dict = {
            "metrics": {
                "mse": {"value": mse, "std": mse_std},
                "mae": {"value": mae, "std": mae_std},
                "R2": {"value": r2, "std": r2_std},
                "rmse": {"value": rmse, "std": rmse_std},
            },
        }

        return report_dict


    def objective_func(
        trial, X_train, y_train, X_val, y_val, task, objective, metric
    ):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): A single trial from Optuna.
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training target data.
            X_val (pd.DataFrame): Validation feature data.
            y_val (pd.Series): Validation target data.
            task (str): Type of machine learning task ('regression' or
                'classification').
            objective (str): Objective function for the XGBoost model.
            metric (str): Metric to optimize.

        Returns:
            float: The computed metric value for the trial.
        """
        # Define hyperparameter search space for the XGBoost model
        param = {
            "silent": 0,
            "verbosity": 0,
            "objective": objective,
            "eta": trial.suggest_float("eta", 0, 1),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "alpha": trial.suggest_float("alpha", 0, 2),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "num_round": trial.suggest_int("num_round", 100, 500),
            "rate_drop": 0.3,
            "tweedie_variance_power": 1.4,
        }

        # Train and evaluate the model based on the task
        if task == "regression":
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = generate_regression_report(y_val, y_pred)["metrics"]
        else:
            model = xgb.XGBClassifier(**param)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = generate_classification_report(y_val, y_pred)["metrics"]

        return score[metric]["value"]


    class MLMetric(BaseTunerMetric):
        """
        Custom metric for ML model evaluation

        Args:
            df_test (pd.DataFrame): The test dataset.
            preprocess (object): Preprocessing steps applied to the data.
            target_column (str): Name of the target column.
            metric (str, optional): Name of the metric to optimize.
            task (str, optional): Type of machine learning task.
            objective (str, optional): Objective function for the XGBoost model.
            objective_type (str, optional): Direction of optimization ('Maximize'
                or 'Minimize').
        """

        def __init__(
            self,
            df_test,
            preprocess,
            target_column,
            metric="f1",
            task="classification",
            objective="binary:logistic",
            objective_type="Maximize",
        ):
            self.df_test = df_test
            self.metric = metric
            self.task = task
            self.preprocess = preprocess
            self.target_column = target_column
            self.objective = objective
            self.direction = MetricDirection[objective_type.upper()]

        def __call__(self, model: Model):
            """
            Evaluates the model using Optuna.

            Args:
                model (Model): The Gretel synthetic model to be evaluated.

            Returns:
                float: The best value of the specified metric.
            """
            # Load and preprocess the training data from the model's artifact
            X_train = pd.read_csv(
                model.get_artifact_link("data_preview"), compression="gzip"
            )
            y_train = X_train.pop(self.target_column)
            X_train = pd.DataFrame(self.preprocess.transform(X_train))

            # Align the columns of the test set to match those of the training set
            X_val = self.df_test.copy()
            y_val = X_val.pop(self.target_column)
            X_val.columns = X_train.columns

            # Create and run an Optuna study for hyperparameter optimization
            study = optuna.create_study(direction=self.direction.value)
            study.optimize(
                lambda trial: objective_func(
                    trial,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    self.task,
                    self.objective,
                    self.metric,
                ),
                n_trials=6,
                n_jobs=2,
            )

            return study.best_value


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # parse config file
    config = json.loads(config)

    # Extract configuration details
    target_column = config["dataset"]["target_column"]
    ml_task = config["ML"]["ml_task"]
    ml_eval_metric = config["ML"]["ml_eval_metric"]
    objective = config["ML"]["objective"]
    objective_type = config["ML"]["objective_type"]
    strategy = config["gretel"]["strategy"]
    generate_factor = config["gretel"]["generate_factor"]
    target_balance = config["gretel"]["target_balance"]
    mode = config["gretel"]["mode"]
    sink_bucket = config["gretel"]["sink_bucket"]

    # Read the training data
    logger.info("Reading train data.")
    data_source = pd.read_csv(f"{input_dir}/train_source.csv")

    # Read the validation data
    logger.info("Reading validation data.")
    data_validation = pd.read_csv(f"{input_dir}/validation.csv")

    # Load the preprocessing model saved earlier
    logger.info("Loading preprocessing model.")
    preprocess = joblib.load(f"{input_dir}/preprocess.joblib")

    if strategy is None:
        # If no strategy is provided, use the source data directly
        logger.info("No Gretel required. Using source data.")
        logger.info("Apply preprocessing transformations.")
        y_train = data_source.pop(target_column)
        train_pre = pd.DataFrame(preprocess.transform(data_source))
        train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1)

        logger.info("Writing out training data.")
        train.to_csv(f"{output_dir}/train.csv", header=True, index=False)

    else:
        # Retrieve Gretel API key from secret
        logger.info("Retrieve Gretel API key from secret.")
        GRETEL_API_KEY = get_secret(gretel_secret, project_number)

        # Configure a Gretel session for synthetic data generation
        logger.info(f"Configuring a {mode} Gretel session.")
        GRETEL_PROJECT_NAME = "vertex-pipelines-gretel-hyptuning"
        gretel = Gretel(
            project_name=GRETEL_PROJECT_NAME,
            api_key=GRETEL_API_KEY,
            validate=True,
            clear=True,
            default_runner=mode,
            artifact_endpoint=f"s3://{sink_bucket}"
            if mode == "hybrid"
            else "cloud",
        )

        if strategy == "balance":
            # Balance the dataset based on the target column
            data_source = naive_upsample(
                data_source,
                target_column=target_column,
                target_balance=target_balance,
            )

        optimization_metric = MLMetric(
            data_validation,
            preprocess,
            target_column,
            metric=ml_eval_metric,
            task=ml_task,
            objective=objective,
            objective_type=objective_type,
        )

        tuner_config = """
          base_config: tabular-actgan

          params:
              batch_size:
                  choices: [500, 1000, 2000]

              epochs:
                  choices: [200, 400, 600, 800, 1200, 1400, 1600]

              generator_lr:
                  log_range: [0.00001, 0.001]

              discriminator_lr:
                  log_range: [0.00001, 0.001]

              generator_dim:
                  choices: [
                      [512, 512, 512, 512],
                      [1024, 1024],
                      [1024, 1024, 1024],
                      [2048, 2048],
                      [2048, 2048, 2048]
                  ]
        """

        def sampler_callback(model_section):
            """Always set discriminator_dim = generator_dim."""
            model_section["params"]["discriminator_dim"] = model_section[
                "params"
            ]["generator_dim"]
            return model_section

        # Running Gretel tuner with the defined configuration
        N_TRIALS = 16
        MAX_JOBS = 4
        tuner_results = gretel.run_tuner(
            tuner_config,
            data_source=data_source,
            n_jobs=MAX_JOBS,
            n_trials=N_TRIALS,
            metric=optimization_metric,
            sampler_callback=sampler_callback,
        )

        # Fetching the best model from Gretel tuner results
        best_model = gretel.fetch_train_job_results(
            tuner_results.best_model_id
        )

        # Writing out Gretel quality scores and report
        logger.info("Writing out Gretel sqs report.")
        report_summary_path = f"{output_dir}/report_quality_scores.txt"
        report_full_path = f"{output_dir}/report_full.json"
        report_synth_data_path = f"{output_dir}/report_synth_data.csv"
        with open(report_full_path, "w") as f:
            f.write(str(best_model.report))
        with open(report_summary_path, "w") as f:
            f.write(json.dumps(best_model.report.quality_scores, indent=4))
        df_synth_report = best_model.fetch_report_synthetic_data()
        df_synth_report.to_csv(
            report_synth_data_path, header=True, index=False
        )

        logger.info("Starting Gretel generation step.")
        # Calculate the number of records to generate based on generate_factor
        RECORDS_TO_GENERATE = int(len(data_source) * generate_factor)
        # Submit a job to generate synthetic data using the best model
        generated = gretel.submit_generate(
            best_model.model_id, num_records=RECORDS_TO_GENERATE
        )

        # Depending on the strategy, replace or augment training data with
        # synthetic data
        logger.info("Augment training data with synthetic data .")
        if strategy == "replace":
            df_train_synth = generated.synthetic_data
        else:
            df_train_synth = pd.concat(
                [data_source, generated.synthetic_data],
                axis=0,
                ignore_index=True,
            )

        # Apply preprocessing transformations to the synthetic data
        logger.info("Apply preprocessing transformations.")
        y_train_synth = df_train_synth.pop(target_column)
        train_synth_pre = pd.DataFrame(preprocess.transform(df_train_synth))
        train_synth = pd.concat(
            [y_train_synth.reset_index(drop=True), train_synth_pre], axis=1
        )

        # Write out the augmented training data to a CSV file
        logger.info("Write out training data augmented with synthetic data.")
        train_synth.to_csv(f"{output_dir}/train.csv", header=True, index=False)

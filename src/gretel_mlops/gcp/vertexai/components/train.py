"""Script to run train ML model."""

from kfp.dsl import InputPath, OutputPath, component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-secret-manager",
        "gretel-client[gcp,tuner]",
        "imblearn",
        "optuna",
        "pandas",
        "scikit-learn",
        "xgboost",
    ],
)
def train_component(
    config: str,
    input_dir: InputPath(),
    gretel_dir: InputPath(),
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
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 mean_absolute_error, mean_squared_error,
                                 precision_recall_curve, r2_score, roc_auc_score)

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

    # Reading training data
    logger.info("Reading train data.")
    X_train = pd.read_csv(f"{gretel_dir}/train.csv")
    y_train = X_train.pop(target_column)

    # Reading validation data
    logger.info("Reading validation data.")
    X_val = pd.read_csv(f"{input_dir}/validation.csv")
    y_val = X_val.pop(target_column)
    # Ensure the columns match between training and validation data
    X_val.columns = X_train.columns

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(direction=objective_type.lower())
    study.optimize(
        lambda trial: objective_func(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            ml_task,
            objective,
            ml_eval_metric,
        ),
        n_trials=6,
        n_jobs=2,
    )

    # Retrain the model with the best hyperparameters
    best_params = study.best_trial.params
    if ml_task == "regression":
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_train, y_train)
    else:
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

    # Save the model and best hyperparameters to files
    logger.info("Saving the final model and best parameters.")
    final_model.save_model(f"{output_dir}/model.bst")
    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(best_params, f)

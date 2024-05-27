"""Evaluation script to assess ML model performance."""

from kfp.dsl import InputPath, OutputPath, component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-secret-manager",
        "gretel-client[gcp,tuner]",
        "git+https://github.com/gretelai/gretel-mlops",
        "imblearn",
        "optuna",
        "pandas",
        "scikit-learn",
        "xgboost",
    ],
)
def evaluate_component(
    config: str,
    input_dir: InputPath(),
    model_dir: InputPath(),
    output_dir: OutputPath(),
):
    import json
    import logging
    import os

    import numpy as np
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

    # Load the xgboost model
    logger.info("Loading xgboost model.")
    model = xgb.Booster()
    model.load_model(f"{model_dir}/model.bst")

    # Load the test data
    logger.info("Reading test data.")
    df = pd.read_csv(f"{input_dir}/test.csv")

    # Separate features and target variable from test data
    logger.info("Preparing test data.")
    y_test = df.pop(target_column)
    X_test = xgb.DMatrix(df.values)

    # Perform predictions using the xgboost model
    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    # Calculate and generate the ML metrics based on the task
    logger.info("Calculating ML metrics.")
    if ml_task == "regression":
        report_dict = generate_regression_report(y_test, predictions)
    else:
        report_dict = generate_classification_report(y_test, predictions)
    logger.info(f"Creating Report: {report_dict}")

    # Write the evaluation report to a file
    logger.info("Writing out evaluation report")
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))

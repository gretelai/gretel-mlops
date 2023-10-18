"""Evaluation script for measuring mean squared error."""
import argparse
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def compute_optimal_f1(y_test, predictions):

    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    selection = ~((precision==0)&(recall==0))
    precision = precision[selection]
    recall = recall[selection]
    thresholds = thresholds[selection[:-1]]

    # Calculate the f-score
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Find the threshold that maximizes the F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)

    # Calculate the corresponding precision and recall at the optimal threshold
    optimal_precision = precision[np.argmax(f1_scores)]
    optimal_recall = recall[np.argmax(f1_scores)]

    # Compute the confusion matrix at the optimal threshold
    predictions_binary = (predictions >= best_threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, predictions_binary)
    
    return best_f1_score, optimal_precision, optimal_recall, conf_matrix


def generate_classification_report(y_test, predictions):
    # Calculate classification metrics  
    f1, precision, recall, conf_matrix = compute_optimal_f1(y_test, predictions)

    # Calculate AUC using sklearn's functions
    roc_auc = roc_auc_score(y_test, predictions)
    pr_auc = average_precision_score(y_test, predictions)

    # Create a report dictionary with both the metrics and their standard deviations
    report_dict = {
        "classification_metrics": {
            "auc": {
                "value": roc_auc,
            },
            "pr_auc": {
                "value": pr_auc,
            },
            "precision": {
                "value": precision,
            },
            "recall": {
                "value": recall,
            },
            "f1": {
                "value": f1,
            },
            "confusion_matrix": {
                "value": conf_matrix.tolist(),
            }
        },
    }
    
    return report_dict


def generate_regression_report(y_test, predictions):
    # Calculate regression metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Calculate standard deviations
    mse_std = np.std((y_test - predictions) ** 2)
    rmse_std = np.std(np.abs(y_test - predictions))
    mae_std = np.std(np.abs(y_test - predictions))
    r2_std = np.std(1 - ((y_test - predictions) ** 2) / ((y_test - np.mean(y_test)) ** 2))

    # Create a report dictionary with both the metrics and their standard deviations
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "std": mse_std
            },
            "mae": {
                "value": mae,
                "std": mae_std
            },
            "R2": {
                "value": r2,
                "std": r2_std
            },
            "rmse": {
                "value": rmse,
                "std": rmse_std
            },
        },
    }
    
    return report_dict


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-task", type=str, default="classification")
    args = parser.parse_args()
    
    ml_task = args.ml_task
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    logger.debug("Calculating ML metrics.")
    if ml_task == "regression":
        report_dict = generate_regression_report(y_test, predictions)
    else:
        report_dict = generate_classification_report(y_test, predictions)
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

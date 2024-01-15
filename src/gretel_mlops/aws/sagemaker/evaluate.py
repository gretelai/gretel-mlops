"""Evaluation script to assess ML model performance."""

import argparse
import json
import logging
import pathlib

import pandas as pd
import xgboost

from utils import (
    generate_regression_report,
    generate_classification_report,
    extract_tar_safely,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    # Initialize logging and argument parsing
    logger.info("Starting evaluation.")
    parser = argparse.ArgumentParser()
    # Define arguments for the script
    parser.add_argument("--ml-task", type=str, default="classification")
    parser.add_argument("--target-column", type=str, required=True)
    args = parser.parse_args()

    # Extract machine learning task and target column from arguments
    ml_task = args.ml_task
    target_column = args.target_column

    # Path to the saved model
    model_path = "/opt/ml/processing/model/model.tar.gz"
    # Safely extract the model
    extract_tar_safely(model_path)

    # Load the xgboost model
    logger.info("Loading xgboost model.")
    model = xgboost.Booster()
    model.load_model("xgboost-model")

    # Load the test data
    logger.info("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path)

    # Separate features and target variable from test data
    logger.info("Preparing test data.")
    y_test = df.pop(target_column)
    X_test = xgboost.DMatrix(df.values)

    # Perform predictions using the xgboost model
    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    # Calculate and generate the ML metrics based on the task
    logger.info("Calculating ML metrics.")
    if ml_task == "regression":
        report_dict = generate_regression_report(y_test, predictions)
    else:
        report_dict = generate_classification_report(y_test, predictions)

    # Log and output the evaluation report
    logger.info(f"Creating Report: {report_dict}")
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write the evaluation report to a file
    logger.info("Writing out evaluation report")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

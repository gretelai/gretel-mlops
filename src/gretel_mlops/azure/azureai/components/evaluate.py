"""Evaluation script to assess ML model performance."""

import argparse
import json
import logging

import pandas as pd
import xgboost as xgb
import yaml
from utils import generate_classification_report, generate_regression_report

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser()
    # Define the command line arguments
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Load configuration from the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract configuration details
    target_column = config["dataset"]["target_column"]
    ml_task = config["ML"]["ml_task"]

    # Extract arguments
    input_dir = args.input_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

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

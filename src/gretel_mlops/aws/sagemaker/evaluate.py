"""Evaluation script for measuring mean squared error."""
import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m", "pip", "install",
    "xgboost"
])

import argparse
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from utils import generate_regression_report, generate_classification_report, is_safe_path

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-task", type=str, default="classification")
    parser.add_argument("--target-column", type=str, required=True)
    args = parser.parse_args()
    
    ml_task = args.ml_task
    target_column = args.target_column
    model_path = "/opt/ml/processing/model/model.tar.gz"
    # with tarfile.open(model_path) as tar:
        # tar.extractall(path=".")

    with tarfile.open(model_path) as tar:
        # Validate members
        safe_members = [m for m in tar.getmembers() if is_safe_path(m.name, m.path)]
        
        # Extract only safe members
        tar.extractall(path=".", members=safe_members)

    logger.debug("Loading xgboost model.")
    model = xgboost.Booster()
    model.load_model('xgboost-model')

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path)

    logger.debug("Reading test data.")
    y_test = df.pop(target_column)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    logger.debug("Calculating ML metrics.")
    if ml_task == "regression":
        report_dict = generate_regression_report(y_test, predictions)
    else:
        report_dict = generate_classification_report(y_test, predictions)
    
    logger.info(f"Creating Report: {report_dict}")
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

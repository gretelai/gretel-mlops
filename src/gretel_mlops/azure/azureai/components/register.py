"""Script to register ML model to model registry."""

import argparse
import json
import logging

import pandas as pd
import yaml
from azureml.core import Model, Run

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser()
    # Define the command line arguments
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--model-display-name", type=str, required=True)
    args = parser.parse_args()

    # Load configuration from the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract configuration details
    ml_metric_threshold = config["ML"]["ml_metric_threshold"]
    ml_eval_metric = config["ML"]["ml_eval_metric"]
    objective_type = config["ML"]["objective_type"]

    # Extract arguments
    eval_dir = args.eval_dir
    model_dir = args.model_dir
    model_display_name = args.model_display_name

    # parse evaluation report
    evaluation_report = json.loads(
        pd.read_json(f"{eval_dir}/evaluation.json").to_json()
    )

    # Check if the evaluation metric exceeds the threshold
    if objective_type.lower() == "maximize":
        meets_condition = (
            evaluation_report["metrics"][ml_eval_metric]["value"]
            >= ml_metric_threshold
        )
    else:
        meets_condition = (
            evaluation_report["metrics"][ml_eval_metric]["value"]
            <= ml_metric_threshold
        )

    # Process based on the condition check
    if meets_condition:
        logger.info(
            f"Evaluation metric {ml_eval_metric} exceeds "
            f"threshold {ml_metric_threshold}."
        )

        # Register the model in Azure ML Workspace
        run = Run.get_context()
        ws = run.experiment.workspace

        # Register the model with specified details and tags
        model_to_register = Model.register(
            workspace=ws,
            model_name=model_display_name,
            model_path=model_dir,
            tags={
                "framework": "XGBoost",
                "evaluation_metric": ml_eval_metric,
                "metric_value": str(
                    evaluation_report["metrics"][ml_eval_metric]["value"]
                ),
            },
            properties={"ml_metric_threshold": str(ml_metric_threshold)},
            model_framework=Model.Framework.CUSTOM,
        )
        logger.info(
            f"Model {model_display_name} registered "
            f"with ID: {model_to_register.id}"
        )

    else:
        logger.info(
            f"Evaluation metric {ml_eval_metric} does not exceed "
            f"threshold {ml_metric_threshold}"
        )

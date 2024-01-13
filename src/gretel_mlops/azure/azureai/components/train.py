"""Script to run train ML model."""

import argparse
import json
import logging

import optuna
import pandas as pd
import xgboost as xgb
import yaml
from utils import objective_func

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser()
    # Define the command line arguments
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--gretel-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Load configuration from the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract configuration details
    target_column = config["dataset"]["target_column"]
    ml_task = config["ML"]["ml_task"]
    ml_eval_metric = config["ML"]["ml_eval_metric"]
    objective = config["ML"]["objective"]
    objective_type = config["ML"]["objective_type"]

    # Extract arguments
    input_dir = args.input_dir
    gretel_dir = args.gretel_dir
    output_dir = args.output_dir

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

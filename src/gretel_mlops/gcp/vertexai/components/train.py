# Train step

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.dsl import component, OutputPath, InputPath
from .utils import objective_func

@component(
    base_image="python:3.9",
    packages_to_install=['xgboost', 'scikit-learn==1.3.0', 'optuna', 'pandas', 'google-cloud-aiplatform'],
)
def train_component(
    config: str,
    input_dir: InputPath(),
    gretel_dir: InputPath(),
    output_dir: OutputPath()

):
    import logging
    import json
    import optuna
    import os
    import warnings

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix,
        mean_squared_error, mean_absolute_error, r2_score
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # parse config file
    config = json.loads(config)

    target_column = config['dataset']['target_column']
    ml_task = config['ML']['ml_task']
    ml_eval_metric = config['ML']['ml_eval_metric']
    objective = config['ML']['objective']
    objective_type = config['ML']['objective_type']

    logger.info("Reading train data.")
    X_train = pd.read_csv(f"{gretel_dir}/train.csv")
    y_train = X_train.pop(target_column)

    logger.info("Reading validation data.")
    X_val = pd.read_csv(f"{input_dir}/validation.csv")
    y_val = X_val.pop(target_column)
    X_val.columns = X_train.columns

    study = optuna.create_study(direction=objective_type.lower())
    study.optimize(lambda trial: objective_func(trial, X_train, y_train, X_val, y_val, ml_task, objective, ml_eval_metric), n_trials=6, n_jobs=2)

    # Retrain the model with the best hyperparameters
    best_params = study.best_trial.params

    if ml_task == 'regression':
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X_train, y_train)
    else:
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)

    # Save the model to a file in the output directory
    final_model.save_model(f"{output_dir}/model.bst")

    # Save the best hyperparameters to a file in the output directory
    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(best_params, f)

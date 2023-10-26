"""Script to run Gretel synthetics on source data."""
import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m", "pip", "install",
    "gretel-client[aws]",
    "git+https://github.com/gretelai/gretel-tuner",
])

import argparse
import logging
import pathlib
import pickle
import json
import boto3
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from gretel_client import Gretel, configure_session
from gretel_client.projects.models import read_model_config
from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll
from gretel_tuner import (
    GretelHyperParameterConfig,
    GretelHyperParameterTuner
)
from utils import MLMetric, get_secret, naive_upsample

warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--generate-factor", type=float, default=1.0)
    parser.add_argument("--target-balance", type=float, default=1.0)
    parser.add_argument("--ml-eval-metric", type=str, default="f1")
    parser.add_argument("--ml-task", type=str, default="classification")
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--objective-type", type=str, default="Maximize")
    args = parser.parse_args()
    
    strategy = args.strategy
    generate_factor = args.generate_factor
    target_balance = args.target_balance
    target_column = args.target_column
    ml_eval_metric = args.ml_eval_metric 
    ml_task = args.ml_task
    objective = args.objective 
    objective_type = args.objective_type   
    
    logger.info("Reading train data.")
    source_path = "/opt/ml/processing/train_source/train.csv"
    data_source = pd.read_csv(source_path)

    logger.info("Reading validation data.")
    validation_path = "/opt/ml/processing/validation/validation.csv"
    data_validation = pd.read_csv(validation_path)
    
    gretel_dir = "/opt/ml/processing/gretel"
    output_dir = "/opt/ml/processing/train"
    pathlib.Path(gretel_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading preprocessing model.")
    preprocess_path = "/opt/ml/processing/preprocess/preprocess.pkl"
    preprocess = pickle.load(open(preprocess_path, "rb"))
    
    if strategy == None:
        logger.info("No Gretel required. Using source data.")
        logger.info("Apply preprocessing transformations.")
        y_train = data_source.pop(target_column)
        train_pre = pd.DataFrame(preprocess.transform(data_source))
        train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1) 
    
        logger.info("Writing out training data.")
        train_data_path = f"{output_dir}/train.csv"
        train.to_csv(train_data_path, header=False, index=False)

    else:
        logger.info("Configuring a Gretel session.")
        GRETEL_API_KEY = get_secret()
        configure_session(
            api_key=GRETEL_API_KEY,
            validate=True,
        )
        GRETEL_PROJECT_NAME = 'sagemaker-pipelines-gretel-hyptuning'
        project = create_or_get_unique_project(name=GRETEL_PROJECT_NAME)

        if strategy == "balance":
            data_source = naive_upsample(data_source, target_column=target_column, target_balance=target_balance)

        artifact_id = project.upload_artifact(data_source)

        config = read_model_config("synthetics/tabular-actgan")
        config["models"][0]["actgan"]["privacy_filters"]["outliers"] = None
        config["models"][0]["actgan"]["privacy_filters"]["similarity"] = None
        config["models"][0]["actgan"]["generate"]["num_records"] = min(25_000, len(data_source))

        optimization_metric = MLMetric(
            data_validation, 
            preprocess, 
            target_column, 
            metric=ml_eval_metric, 
            task=ml_task, 
            objective=objective, 
            objective_type=objective_type
        )
        tuner_config = GretelHyperParameterConfig(
            project=project,
            artifact_id=artifact_id,
            epoch_choices=[200, 400, 600, 800, 1200, 1400, 1600],
            batch_size_choices=[500, 1000, 2000],
            base_config=config,
            metric=optimization_metric,
        )

        tuner = GretelHyperParameterTuner(tuner_config)
        N_TRIALS = 16
        MAX_JOBS = 4

        print(f"Running optuna with {N_TRIALS} trials and {MAX_JOBS}")
        tuner_results = tuner.run(n_trials=N_TRIALS, n_jobs=min(N_TRIALS, MAX_JOBS))
        best_config = tuner_results.best_config
        print(best_config)
        best_config_path = f"{gretel_dir}/best_config.json"
        with open(best_config_path, "w") as f:
            f.write(json.dumps(best_config, indent=4))

        logger.info("Starting Gretel training step.")
        gretel = Gretel(api_key=GRETEL_API_KEY, validate=True)
        trained = gretel.submit_train(
            base_config="tabular-actgan",
            data_source=data_source,
            params=best_config['models'][0]['actgan']['params'],
            privacy_filters={"similarity": None, "outliers": None},
        )

        logger.info("Writing out Gretel sqs report.")
        report_summary_path = f"{gretel_dir}/report_quality_scores.txt"
        report_full_path = f"{gretel_dir}/report_full.json"
        report_synth_data_path = f"{gretel_dir}/report_synth_data.csv"
        with open(report_full_path, 'w') as f:
            f.write(str(trained.report))
        with open(report_summary_path, "w") as f:
            f.write(json.dumps(trained.report.quality_scores, indent=4))
        df_synth_report = trained.fetch_report_synthetic_data()
        df_synth_report.to_csv(report_synth_data_path, header=True, index=False)

        logger.info("Starting Gretel generation step.")
        RECORDS_TO_GENERATE = int(len(data_source) * generate_factor)
        generated = gretel.submit_generate(trained.model_id, num_records=RECORDS_TO_GENERATE)

        logger.info("Augment training data with synthetic data .")
        df_train_synth = pd.concat([data_source, generated.synthetic_data], axis=0, ignore_index=True)

        logger.info("Apply preprocessing transformations.")
        y_train_synth = df_train_synth.pop(target_column)
        train_synth_pre = pd.DataFrame(preprocess.transform(df_train_synth))
        train_synth = pd.concat([y_train_synth.reset_index(drop=True), train_synth_pre], axis=1) 

        logger.info("Write out training data augmented with synthetic data.")
        train_synth_data_path = f"{output_dir}/train.csv"
        train_synth.to_csv(train_synth_data_path, header=False, index=False)

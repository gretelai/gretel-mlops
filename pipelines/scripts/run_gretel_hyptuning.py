"""Script to run Gretel synthetics on source data."""
import argparse
import logging
import pathlib
import pickle
import json
import boto3
import os

import numpy as np
import pandas as pd
from datasets import datasets

from gretel_client import Gretel, configure_session
from gretel_client.projects.models import read_model_config
from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll
from botocore.exceptions import ClientError
from gretel_tuner import (
    GretelSQSMetric,
    GretelHyperParameterConfig,
    GretelHyperParameterTuner
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import sklearn
def get_secret():

    secret_name = "prod/Gretel/ApiKey"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])

    return secret["gretelApiKey"]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-factor", type=float, default=1.0)
    parser.add_argument("--label-column-name", type=str, required=True)
    args = parser.parse_args()
    
    logger.info("Reading test data.")
    source_path = "/opt/ml/processing/train_source/train.csv"
    label_column_name = args.label_column_name
    data_source = pd.read_csv(source_path)

    output_dir = "/opt/ml/processing/gretel"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading preprocessing model.")
    preprocess_path = "/opt/ml/processing/preprocess/preprocess.pkl"
    preprocess = pickle.load(open(preprocess_path, "rb"))
    
    logger.info("Configuring a Gretel session.")
    GRETEL_API_KEY = get_secret()
    configure_session(
        api_key=GRETEL_API_KEY,
        validate=True,
    )
    GRETEL_PROJECT_NAME = 'sagemaker-pipelines-gretel-hyptuning'
    project = create_or_get_unique_project(name=GRETEL_PROJECT_NAME)
    artifact_id = project.upload_artifact(data_source)
    
    config = read_model_config("synthetics/tabular-actgan")
    config["models"][0]["actgan"]["privacy_filters"]["outliers"] = None
    config["models"][0]["actgan"]["privacy_filters"]["similarity"] = None
    config["models"][0]["actgan"]["generate"]["num_records"] = min(50000, len(data_source))

    optimization_metric = GretelSQSMetric()
    tuner_config = GretelHyperParameterConfig(
        project=project,
        artifact_id=artifact_id,
        epoch_choices=[200, 400, 600, 800],
        batch_size_choices=[1000, 2000],
        base_config=config,
        metric=optimization_metric,
    )

    tuner = GretelHyperParameterTuner(tuner_config)
    N_TRIALS = 1#16
    MAX_JOBS = 1#4

    print(f"Running optuna with {N_TRIALS} trials and {MAX_JOBS}")
    tuner_results = tuner.run(n_trials=N_TRIALS, n_jobs=min(N_TRIALS, MAX_JOBS))
    best_config = tuner_results.best_config
    best_config_path = f"{output_dir}/best_config.json"
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
    report_summary_path = f"{output_dir}/report_quality_scores.txt"
    report_full_path = f"{output_dir}/report_full.json"
    report_synth_data_path = f"{output_dir}/report_synth_data.csv"
    with open(report_full_path, 'w') as f:
        f.write(str(trained.report))
    with open(report_summary_path, "w") as f:
        f.write(json.dumps(trained.report.quality_scores, indent=4))
    df_synth_report = trained.fetch_report_synthetic_data()
    df_synth_report.to_csv(report_synth_data_path, header=True, index=False)
    
    logger.info("Starting Gretel generation step.")
    generate_factor = args.generate_factor
    RECORDS_TO_GENERATE = int(len(data_source) * generate_factor)
    generated = gretel.submit_generate(trained.model_id, num_records=RECORDS_TO_GENERATE)
    df_synth = generated.synthetic_data
    
    logger.info("Augment training data with synthetic data .")
    df_train_synth = pd.concat([data_source, df_synth], axis=0, ignore_index=True)
    df_train_synth = data_source
    
    logger.info("Apply preprocessing transformations.")
    y_train_synth = df_train_synth.pop(label_column_name)
    train_synth_pre = pd.DataFrame(preprocess.transform(df_train_synth))
    train_synth = pd.concat([y_train_synth.reset_index(drop=True), train_synth_pre], axis=1) 

    logger.info("Write out training data augmented with synthetic data.")
    train_synth_dir = "/opt/ml/processing/train_synth"
    pathlib.Path(train_synth_dir).mkdir(parents=True, exist_ok=True)
    train_synth_data_path = f"{train_synth_dir}/train.csv"
    train_synth.to_csv(train_synth_data_path, header=False, index=False)
"""Script to run Gretel synthetics on source data."""
import argparse
import logging
import pathlib
import json
import boto3

import numpy as np
import pandas as pd
from gretel_client import Gretel
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


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
    logger.debug("Starting Gretel Synthetics.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-factor", type=float, default=1.0)
    args = parser.parse_args()
    
    logger.debug("Reading test data.")
    source_path = "/opt/ml/processing/train/train.csv"
    data_source = pd.read_csv(source_path, header=None)

    output_dir = "/opt/ml/processing/gretel"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Configure a Gretel session
    GRETEL_API_KEY = get_secret()
    gretel = Gretel(api_key=GRETEL_API_KEY, validate=True)
    
    # Train a Generative Model
    trained = gretel.submit_train("tabular-actgan", data_source=data_source)
    logger.info("Gretel training step completed")

    # Evaluate Synthetic data Quality
    logger.info("Writing out Gretel sqs report")
    report_summary_path = f"{output_dir}/report_quality_scores.txt"
    report_full_path = f"{output_dir}/report_full.json"
    
    with open(report_full_path, 'w') as f:
        f.write(str(trained.report))
    with open(report_summary_path, "w") as f:
        f.write(json.dumps(trained.report.quality_scores, indent=4))
    
    # Generate Synthetic data
    generate_factor = args.generate_factor
    RECORDS_TO_GENERATE = int(len(data_source) * generate_factor)
    generated = gretel.submit_generate(trained.model_id, num_records=RECORDS_TO_GENERATE)
    logger.info("Gretel generation step completed")
    
    # Write out synthetic data
    synth_data_path = f"{output_dir}/synth.csv"
    generated.synthetic_data.to_csv(synth_data_path)
    

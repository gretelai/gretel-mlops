"""Script to run Gretel synthetics on source data."""

import argparse
import json
import logging
import warnings

import joblib
import pandas as pd
import yaml
from gretel_client import Gretel
from utils import MLMetric, naive_upsample

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser()
    # Define the command line arguments
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gretel-api-key", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
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
    strategy = config["gretel"]["strategy"]
    generate_factor = config["gretel"]["generate_factor"]
    target_balance = config["gretel"]["target_balance"]
    mode = config["gretel"]["mode"]
    sink_bucket = config["gretel"]["sink_bucket"]

    # Extract arguments
    gretel_api_key = args.gretel_api_key
    input_dir = args.input_dir
    output_dir = args.output_dir

    logger.info("Reading train data.")
    # Define the path to the source training data and read it
    data_source = pd.read_csv(f"{input_dir}/train_source.csv")

    logger.info("Reading validation data.")
    # Define the path to the validation data and read it
    data_validation = pd.read_csv(f"{input_dir}/validation.csv")

    logger.info("Loading preprocessing model.")
    # Load the preprocessing model saved earlier
    preprocess = joblib.load(f"{input_dir}/preprocess.joblib")

    if strategy is None:
        # If no strategy is provided, use the source data directly
        logger.info("No Gretel required. Using source data.")
        logger.info("Apply preprocessing transformations.")
        y_train = data_source.pop(target_column)
        train_pre = pd.DataFrame(preprocess.transform(data_source))
        train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1)

        logger.info("Writing out training data.")
        train_data_path = f"{output_dir}/train.csv"
        train.to_csv(train_data_path, header=False, index=False)

    else:
        # Configure a Gretel session for synthetic data generation
        logger.info(f"Configuring a {mode} Gretel session.")

        GRETEL_PROJECT_NAME = "azureai-pipelines-gretel-hyptuning"
        gretel = Gretel(
            project_name=GRETEL_PROJECT_NAME,
            api_key=gretel_api_key,
            validate=True,
            clear=True,
            default_runner=mode,
            artifact_endpoint=f"s3://{sink_bucket}"
            if mode == "hybrid"
            else "cloud",
        )

        if strategy == "balance":
            # Balance the dataset based on the target column
            data_source = naive_upsample(
                data_source,
                target_column=target_column,
                target_balance=target_balance,
            )

        optimization_metric = MLMetric(
            data_validation,
            preprocess,
            target_column,
            metric=ml_eval_metric,
            task=ml_task,
            objective=objective,
            objective_type=objective_type,
        )

        tuner_config = """
            base_config: tabular-actgan

            params:
                batch_size:
                    choices: [500, 1000, 2000]

                epochs:
                    choices: [200, 400, 600, 800, 1200, 1400, 1600]

                generator_lr:
                    log_range: [0.00001, 0.001]

                discriminator_lr:
                    log_range: [0.00001, 0.001]

                generator_dim:
                    choices: [
                        [512, 512, 512, 512],
                        [1024, 1024],
                        [1024, 1024, 1024],
                        [2048, 2048],
                        [2048, 2048, 2048]
                    ]
        """

        def sampler_callback(model_section):
            """Always set discriminator_dim = generator_dim."""
            model_section["params"]["discriminator_dim"] = model_section[
                "params"
            ]["generator_dim"]
            return model_section

        # Running Gretel tuner with the defined configuration
        N_TRIALS = 16
        MAX_JOBS = 4
        tuner_results = gretel.run_tuner(
            tuner_config,
            data_source=data_source,
            n_jobs=MAX_JOBS,
            n_trials=N_TRIALS,
            metric=optimization_metric,
            sampler_callback=sampler_callback,
        )

        # Fetching the best model from Gretel tuner results
        best_model = gretel.fetch_train_job_results(
            tuner_results.best_model_id
        )

        # Writing out Gretel quality scores and report
        logger.info("Writing out Gretel quality scores and report.")
        report_summary_path = f"{output_dir}/report_quality_scores.txt"
        report_full_path = f"{output_dir}/report_full.json"
        report_synth_data_path = f"{output_dir}/report_synth_data.csv"
        with open(report_full_path, "w") as f:
            f.write(str(best_model.report))
        with open(report_summary_path, "w") as f:
            f.write(json.dumps(best_model.report.quality_scores, indent=4))
        df_synth_report = best_model.fetch_report_synthetic_data()
        df_synth_report.to_csv(
            report_synth_data_path, header=True, index=False
        )

        logger.info("Starting Gretel generation step.")
        # Calculate the number of records to generate based on generate_factor
        RECORDS_TO_GENERATE = int(len(data_source) * generate_factor)
        # Submit a job to generate synthetic data using the best model
        generated = gretel.submit_generate(
            best_model.model_id, num_records=RECORDS_TO_GENERATE
        )

        logger.info("Augment training data with synthetic data.")
        # Depending on the strategy, replace or augment training data with
        # synthetic data
        if strategy == "replace":
            df_train_synth = generated.synthetic_data
        else:
            df_train_synth = pd.concat(
                [data_source, generated.synthetic_data],
                axis=0,
                ignore_index=True,
            )

        logger.info("Apply preprocessing transformations.")
        # Apply preprocessing transformations to the synthetic data
        y_train_synth = df_train_synth.pop(target_column)
        train_synth_pre = pd.DataFrame(preprocess.transform(df_train_synth))
        train_synth = pd.concat(
            [y_train_synth.reset_index(drop=True), train_synth_pre], axis=1
        )

        logger.info("Write out training data augmented with synthetic data.")
        # Write out the augmented training data to a CSV file
        train_synth_data_path = f"{output_dir}/train.csv"
        train_synth.to_csv(train_synth_data_path, header=True, index=False)

# Gretel step

from kfp.dsl import component, OutputPath, InputPath
        
@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-secret-manager",
        "gretel-client[gcp]",
        "git+https://github.com/gretelai/gretel-tuner",
        "git+https://github.com/gretelai/gretel-mlops",
        "imblearn",
        "optuna",
        "xgboost",
    ],
)
def gretel_component(
    config: str,
    gretel_api_key: str,
    input_dir: InputPath(),
    output_dir: OutputPath()
):
    import logging
    import os
    import json
    import pandas as pd
    from joblib import load
    from imblearn.over_sampling import RandomOverSampler
    from gretel_client import Gretel
    from gretel_client.projects.models import read_model_config
    from gretel_tuner import (
        GretelHyperParameterConfig,
        GretelHyperParameterTuner
    )
    from gretel_mlops.gcp.vertexai.utils import (
      MLMetric,
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    def naive_upsample(df, target_column, target_balance=1.0):

        over_sampler = RandomOverSampler(sampling_strategy=target_balance)
        y = df.pop(target_column)
        df_resampled, y_resampled = over_sampler.fit_resample(df, y)
        df_resampled[target_column] = y_resampled

        return df_resampled

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # parse config file
    config = json.loads(config)

    target_column = config['dataset']['target_column']
    ml_task = config['ML']['ml_task']
    ml_eval_metric = config['ML']['ml_eval_metric']
    objective = config['ML']['objective']
    objective_type = config['ML']['objective_type']
    strategy = config['gretel']['strategy']
    generate_factor = config['gretel']['generate_factor']
    target_balance = config['gretel']['target_balance']
    mode = config['gretel']['mode']
    sink_bucket = config['gretel']['sink_bucket']

    logger.info("Reading train data.")
    data_source = pd.read_csv(f"{input_dir}/train_source.csv")

    logger.info("Reading validation data.")
    data_validation = pd.read_csv(f"{input_dir}/validation.csv")

    logger.info("Loading preprocessing model.")
    preprocess = load(f"{input_dir}/preprocess.joblib")

    if strategy == None:
        logger.info("No Gretel required. Using source data.")
        logger.info("Apply preprocessing transformations.")
        y_train = data_source.pop(target_column)
        train_pre = pd.DataFrame(preprocess.transform(data_source))
        train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1)

        logger.info("Writing out training data.")
        train.to_csv(f"{output_dir}/train.csv", header=True, index=False)

    else:

      logger.info(f"Configuring a {mode} Gretel session.")

      GRETEL_PROJECT_NAME = 'vertex-pipelines-gretel-hyptuning'
      gretel = Gretel(
        project_name=GRETEL_PROJECT_NAME,
        api_key=gretel_api_key,
        validate=True,
        clear=True,
        default_runner=mode,
        artifact_endpoint=f"s3://{sink_bucket}" if mode == "hybrid" else "cloud"
      )

      if strategy == "balance":
          data_source = naive_upsample(data_source, target_column=target_column, target_balance=target_balance)

      project = gretel.get_project()
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
          epoch_choices=[20],#0, 400, 600, 800, 1200, 1400, 1600],
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
      best_config_path = f"{output_dir}/best_config.json"
      with open(best_config_path, "w") as f:
          f.write(json.dumps(best_config, indent=4))

      logger.info("Starting Gretel training step.")
      gretel = Gretel(api_key=gretel_api_key, validate=True)
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
      RECORDS_TO_GENERATE = int(len(data_source) * generate_factor)
      generated = gretel.submit_generate(trained.model_id, num_records=RECORDS_TO_GENERATE)

      logger.info("Augment training data with synthetic data .")
      if strategy == "replace":
          df_train_synth = generated.synthetic_data
      else:
          df_train_synth = pd.concat([data_source, generated.synthetic_data], axis=0, ignore_index=True)

      logger.info("Apply preprocessing transformations.")
      y_train_synth = df_train_synth.pop(target_column)
      train_synth_pre = pd.DataFrame(preprocess.transform(df_train_synth))
      train_synth = pd.concat([y_train_synth.reset_index(drop=True), train_synth_pre], axis=1)

      logger.info("Write out training data augmented with synthetic data.")
      train_synth.to_csv(f"{output_dir}/train.csv", header=True, index=False)

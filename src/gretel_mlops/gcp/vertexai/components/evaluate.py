# Evaluate step

from kfp.dsl import component, OutputPath, InputPath

@component(
    base_image="python:3.10",
    packages_to_install=[
      'xgboost', 
      'scikit-learn==1.3.0', 
      'optuna', 
      'pandas', 
      'google-cloud-aiplatform',
      "git+https://github.com/gretelai/gretel-mlops",
    ],
)
def evaluate_component(
    config: str,
    input_dir: InputPath(),
    model_dir: InputPath(),
    output_dir: OutputPath(),
):
    import logging
    import os
    import json
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from gretel_mlops.gcp.vertexai.components.utils import (
      generate_regression_report,
      generate_classification_report
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

    logger.info("Loading xgboost model.")
    model = xgb.Booster()
    model.load_model(f"{model_dir}/model.bst")

    logger.info("Reading test data.")
    df = pd.read_csv(f"{input_dir}/test.csv")

    logger.info("Reading test data.")
    y_test = df.pop(target_column)
    X_test = xgb.DMatrix(df.values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    logger.info("Calculating ML metrics.")
    if ml_task == "regression":
        report_dict = generate_regression_report(y_test, predictions)
    else:
        report_dict = generate_classification_report(y_test, predictions)

    logger.info("Writing out evaluation report")
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
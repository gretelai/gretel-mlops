"""Example workflow pipeline script for abalone pipeline.

                                                                  . -ModelStep
                                                                 .
    Process -> Synthetic Data -> Train -> Evaluate -> Condition .
                                                                 .
                                                                  . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    FrameworkProcessor,
)
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.workflow.conditions import (
    ConditionLessThanOrEqualTo,
    ConditionGreaterThan,
)
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TuningStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """
    Creates and returns a SageMaker client for a specific AWS region.

    Args:
        region (str): AWS region to start the session.

    Returns:
        boto3.client: A SageMaker client instance.
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """
    Creates and returns a SageMaker session.

    Args:
        region (str): AWS region to start the session.
        default_bucket (str): Name of the S3 bucket for storing artifacts.

    Returns:
        sagemaker.session.Session: A SageMaker session instance.
    """
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """
    Creates and returns a SageMaker Pipeline session.

    Args:
        region (str): AWS region to start the session.
        default_bucket (str): Name of the S3 bucket for storing artifacts.

    Returns:
        PipelineSession: A SageMaker PipelineSession instance.
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="GretelPackageGroup",
    pipeline_name="GretelPipeline",
    base_job_prefix="Gretel",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    gretel_secret=None,
    config=None,
):
    """
    Constructs and returns a SageMaker ML Pipeline for tabular data processing.

    This pipeline includes steps for data preprocessing, synthetic data
     generation, model training, evaluation, and conditional model registration
     based on evaluation results.

    Args:
        region (str): AWS region where the pipeline will be created.
        sagemaker_project_name (str, optional): Name of the SageMaker project.
        role (str, optional): IAM role for pipeline execution.
        default_bucket (str, optional): S3 bucket for storing artifacts.
        model_package_group_name (str, optional): Name for the model
          package group.
        pipeline_name (str, optional): Name of the pipeline.
        base_job_prefix (str, optional): Prefix for naming SageMaker jobs.
        processing_instance_type (str, optional): EC2 instance type for
          processing jobs.
        training_instance_type (str, optional): EC2 instance type for
          training jobs.
        gretel_secret (str, optional): Secret key for Gretel API.
        config (dict, optional): Configuration parameters for the pipeline.

    Returns:
        Pipeline: A SageMaker ML Pipeline instance.
    """

    # Create a SageMaker session for pipeline operations
    sagemaker_session = get_session(region, default_bucket)
    # If no role is provided, use the default execution role
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Create a Pipeline session for executing the pipeline steps
    pipeline_session = get_pipeline_session(region, default_bucket)

    # Define pipeline parameters for execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    dataset = config["dataset"]["name"]
    dataset_name = ParameterString(
        name="InputDatasetName",
        default_value=dataset,
    )

    # Config parameters
    train_path = config["dataset"]["train_path"]
    validation_path = config["dataset"]["validation_path"]
    test_path = config["dataset"]["test_path"]
    target_column = config["dataset"]["target_column"]
    drop_columns = config["dataset"]["drop_columns"]
    ml_task = config["ML"]["ml_task"]
    ml_eval_metric = config["ML"]["ml_eval_metric"]
    ml_eval_metric = config["ML"]["ml_eval_metric"]
    objective = config["ML"]["objective"]
    objective_type = config["ML"]["objective_type"]
    ml_metric_threshold = config["ML"]["ml_metric_threshold"]
    strategy = config["gretel"]["strategy"]
    generate_factor = config["gretel"]["generate_factor"]
    target_balance = config["gretel"]["target_balance"]
    mode = config["gretel"]["mode"]
    sink_bucket = config["gretel"]["sink_bucket"]

    # Configure arguments for the preprocessing step
    arguments = [
        "--train-path",
        train_path,
        "--target-column",
        target_column,
        "--ml-task",
        ml_task,
    ]
    if validation_path:
        arguments += [
            "--validation-path",
            validation_path,
        ]
    if test_path:
        arguments += [
            "--test-path",
            test_path,
        ]
    if drop_columns:
        arguments += [
            "--drop-columns",
            drop_columns,
        ]

    # Define the preprocessing step using a FrameworkProcessor
    script_preprocess = FrameworkProcessor(
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
        estimator_cls=PyTorch,
        framework_version="2.0",
        py_version="py310",
    )
    step_args = script_preprocess.run(
        outputs=[
            ProcessingOutput(
                output_name="train", source="/opt/ml/processing/train"
            ),
            ProcessingOutput(
                output_name="train_source",
                source="/opt/ml/processing/train_source",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
            ),
            ProcessingOutput(
                output_name="test", source="/opt/ml/processing/test"
            ),
            ProcessingOutput(
                output_name="preprocess",
                source="/opt/ml/processing/preprocess",
            ),
        ],
        code="preprocess.py",
        source_dir=BASE_DIR,
        arguments=arguments,
    )
    step_process = ProcessingStep(
        name="PreprocessData",
        step_args=step_args,
    )

    # Define the Gretel step for synthetic data generation
    script_gretel = FrameworkProcessor(
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-gretel-synthetics",
        sagemaker_session=pipeline_session,
        role=role,
        estimator_cls=PyTorch,
        framework_version="2.0",
        py_version="py310",
    )
    arguments = [
        "--target-column",
        target_column,
        "--gretel-secret",
        gretel_secret,
        "--region",
        region,
    ]
    if strategy:
        arguments += [
            "--strategy",
            strategy,
            "--ml-eval-metric",
            ml_eval_metric,
            "--generate-factor",
            str(generate_factor),
            "--ml-task",
            ml_task,
            "--objective",
            objective,
            "--objective-type",
            objective_type,
            "--mode",
            mode,
        ]
    if target_balance:
        arguments += [
            "--target-balance",
            str(target_balance),
        ]
    if sink_bucket:
        arguments += [
            "--sink-bucket",
            sink_bucket,
        ]
    step_args = script_gretel.run(
        inputs=[
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train_source"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/train_source",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "preprocess"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/preprocess",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="gretel", source="/opt/ml/processing/gretel"
            ),
            ProcessingOutput(
                output_name="train", source="/opt/ml/processing/train"
            ),
        ],
        code="run_gretel.py",
        source_dir=BASE_DIR,
        arguments=arguments,
    )
    step_gretel = ProcessingStep(
        name="GretelSynthetics",
        step_args=step_args,
    )

    # Define the model training step
    model_path = (
        f"s3://{sagemaker_session.default_bucket()}/"
        f"{base_job_prefix}/ModelTrain"
    )
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    fixed_hyperparameters = {
        "eval_metric": ml_eval_metric,
        "objective": objective,
        "rate_drop": "0.3",
        "tweedie_variance_power": "1.4",
    }
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        hyperparameters=fixed_hyperparameters,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/model-train",
        sagemaker_session=pipeline_session,
        role=role,
    )
    hyperparameter_ranges = {
        "eta": ContinuousParameter(0, 1),
        "min_child_weight": ContinuousParameter(1, 10),
        "alpha": ContinuousParameter(0, 2),
        "max_depth": IntegerParameter(1, 10),
        "num_round": IntegerParameter(100, 500),
    }
    objective_metric_name = f"validation:{ml_eval_metric}"
    xgb_tuner = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=16,
        max_parallel_jobs=4,
        objective_type=objective_type,
    )

    step_args = xgb_tuner.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_gretel.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    step_train = TuningStep(
        name="TrainModel",
        step_args=step_args,
    )

    # Define the model evaluation step
    script_eval = FrameworkProcessor(
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-model-eval",
        sagemaker_session=pipeline_session,
        role=role,
        estimator_cls=PyTorch,
        framework_version="2.0.1",
        py_version="py310",
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.get_top_model_s3_uri(
                    top_k=0,
                    s3_bucket=sagemaker_session.default_bucket(),
                    prefix=f"{base_job_prefix}/ModelTrain",
                ),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            ),
        ],
        code="evaluate.py",
        source_dir=BASE_DIR,
        arguments=["--ml-task", ml_task, "--target-column", target_column],
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )

    # Define the model registration step
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0][
                    "S3Output"
                ]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_train.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=sagemaker_session.default_bucket(),
            prefix=f"{base_job_prefix}/ModelTrain",
        ),
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="Model",
        step_args=step_args,
    )

    # Define the condition step for evaluating model quality
    left_condition = JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path=f"metrics.{ml_eval_metric}.value",
    )
    if objective_type == "Minimize":
        cond = ConditionLessThanOrEqualTo(
            left=left_condition,
            right=ml_metric_threshold,
        )
    else:
        cond = ConditionGreaterThan(
            left=left_condition,
            right=ml_metric_threshold,
        )

    step_cond = ConditionStep(
        name="CheckEvaluation",
        conditions=[cond],
        if_steps=[step_register],
        else_steps=[],
    )

    # Initialize and return the pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            dataset_name,
        ],
        steps=[step_process, step_gretel, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline

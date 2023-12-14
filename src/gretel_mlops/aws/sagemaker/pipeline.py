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
    ScriptProcessor,
    FrameworkProcessor,
)
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
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
    TrainingStep,
    TuningStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
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
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

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
    config=None
):
    """Gets a SageMaker ML Pipeline instance working with on a tabular dataset.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    dataset = config['dataset']['name']
    dataset_name = ParameterString(
        name="InputDatasetName",
        default_value=dataset,
    )

    # config parameters
    train_path = config['dataset']['train_path']
    validation_path = config['dataset']['validation_path']
    test_path = config['dataset']['test_path']
    target_column = config['dataset']['target_column']
    drop_columns = config['dataset']['drop_columns']
    ml_task = config['ML']['ml_task']
    ml_eval_metric = config['ML']['ml_eval_metric']
    ml_eval_metric = config['ML']['ml_eval_metric']
    objective = config['ML']['objective']
    objective_type = config['ML']['objective_type']
    ml_metric_threshold = config['ML']['ml_metric_threshold']
    strategy = config['gretel']['strategy']
    generate_factor = config['gretel']['generate_factor']
    target_balance = config['gretel']['target_balance']
    mode = config['gretel']['mode']
    sink_bucket = config['gretel']['sink_bucket']

    arguments = [
        "--train-path", train_path,
        "--target-column", target_column,
        "--ml-task", ml_task,
    ]
    if validation_path:
        arguments += [
            "--validation-path", validation_path,
        ]
    if test_path:
        arguments += [
            "--test-path", test_path,
        ]
    if drop_columns:
        arguments += [
            "--drop-columns", drop_columns,
        ]
    
    # processing step for feature engineering
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
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="train_source", source="/opt/ml/processing/train_source"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ProcessingOutput(output_name="preprocess", source="/opt/ml/processing/preprocess"),
        ],
        code="preprocess.py",
        source_dir=BASE_DIR,
        arguments=arguments,
    )
    step_process = ProcessingStep(
        name="PreprocessData",
        step_args=step_args,
    )


    # gretel step for synthetic data generation       
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
        "--target-column", target_column,
        "--gretel-secret", gretel_secret,
        "--region", region,
    ]
    if strategy:
        arguments += [
            "--strategy", strategy,
            "--ml-eval-metric", ml_eval_metric,
            "--generate-factor", str(generate_factor),
            "--ml-task", ml_task,
            "--objective", objective,
            "--objective-type", objective_type,
            "--mode", mode,
        ]
    if target_balance:
        arguments += [
            "--target-balance", str(target_balance),
        ]
    if sink_bucket:
        arguments += [
            "--sink-bucket", sink_bucket,
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
            ProcessingOutput(output_name="gretel", source="/opt/ml/processing/gretel"),
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ],
        code="run_gretel.py",
        source_dir=BASE_DIR,
        arguments=arguments,
    )
    step_gretel = ProcessingStep(
        name="GretelSynthetics",
        step_args=step_args,
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/ModelTrain"
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
        "tweedie_variance_power": "1.4"
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
        objective_type=objective_type
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


    # processing step for evaluation
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
                source=step_train.get_top_model_s3_uri(top_k=0,s3_bucket=sagemaker_session.default_bucket(),prefix=f"{base_job_prefix}/ModelTrain"),
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
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code="evaluate.py",
        source_dir=BASE_DIR,
        arguments=[
            "--ml-task", ml_task,
            "--target-column", target_column
        ],
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

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model = Model(
        image_uri=image_uri,
        model_data=step_train.get_top_model_s3_uri(top_k=0,s3_bucket=sagemaker_session.default_bucket(),prefix=f"{base_job_prefix}/ModelTrain"),
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

    # condition step for evaluating model quality and branching execution
    left_condition = JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path=f"metrics.{ml_eval_metric}.value"
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

    # pipeline instance
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
    return pipeline, BASE_DIR

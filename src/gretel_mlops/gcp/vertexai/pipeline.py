# Pipeline

import json
import pandas as pd
import google.cloud.aiplatform as aip
from kfp import dsl
from .components import (
    preprocess_component,
    gretel_component,
    train_component,
    evaluate_component,
    register_component
)

##
def create_pipeline(
    pipeline_name, 
    pipeline_root,
    model_display_name,
    model_image_uri,
    project_id,
    region,
    gretel_api_key,
    config
):

    @dsl.pipeline(
      name=pipeline_name,
      pipeline_root=pipeline_root,
    )
    def pipeline():
        preprocess_op = preprocess_component(
          config=config
        )
    
        gretel_op = gretel_component(
            config=config,
            project_number=project_number,
            input_dir=preprocess_op.output,
            gretel_api_key=gretel_api_key
            )

        train_op = train_component(
            config=config,
            input_dir=preprocess_op.output,
            gretel_dir=gretel_op.output
            )

        eval_op = evaluate_component(
            config=config,
            input_dir=preprocess_op.output,
            model_dir=train_op.output
            )

        register_op = register_component(
            config=config,
            model_display_name=model_display_name,
            model_image_uri=model_image_uri,
            project=project_id,
            location=region,
            eval_dir=eval_op.output,
            model_dir=train_op.output
            )

    return pipeline

##
def get_pipeline_job_result(
  job_name: str, 
  project: str, 
  location: str
):

    aip.init(project=project, location=location)

    # Get the job using the job name
    pipeline_job = aip.PipelineJob.get(resource_name=job_name)

    # Ensure the job is completed
    if str(pipeline_job.state) != "PipelineState.PIPELINE_STATE_SUCCEEDED":
        print(f"The job state is {pipeline_job.state}, please wait until it's completed.")
        return None

    # get evaluation task
    tasks_dict = {task.task_name: task for task in pipeline_job.task_details}
    eval_task = tasks_dict.get('evaluate-component')

    # retrieve evaluation report
    evaluation_path = f"{dict(eval_task.outputs)['output_dir'].artifacts[0].uri}/evaluation.json"
    evaluation_report = json.loads(pd.read_json(evaluation_path).to_json())

    return json.dumps(evaluation_report, indent=4)
import getpass
import yaml

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azureml.core import Datastore


def create_ml_client(subscription_id, resource_group, workspace_name):
    """
    Creates an MLClient instance for Azure Machine Learning operations.

    Args:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace_name (str): Azure ML workspace name.

    Returns:
        MLClient: An instance of Azure MLClient.
    """
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential, subscription_id, resource_group, workspace_name
    )
    return ml_client


def create_asset_from_config(
    ml_client,
    workspace,
    config_dict,
    asset_name,
    asset_version="v0",
    datastore_name="greteldatastore",
):
    """
    Create or update a data asset based on a provided configuration dictionary
    in an Azure ML workspace. The function checks if the specified datastore
    exists, and if not, prompts for credentials to create a new Azure Blob
    Storage datastore. It then attempts to retrieve an existing data asset
    with the given name and version; if not found, it creates a new data asset.

    Args:
        ml_client: The Azure Machine Learning client object used for asset
            operations.
        workspace: The Azure ML Workspace object where the asset will be
            created or updated.
        config_dict: A dictionary containing configuration information for the
            data asset.
        asset_name: The name of the data asset to be created or updated.
        asset_version: The version of the data asset (default is 'v0').
        datastore_name: The name of the datastore to be used (default is
            'greteldatastore').

    Returns:
        The created or updated data asset object.

    Raises:
        Exception: If there is an error in creating or updating the data asset.
    """
    # Create a YAML file from the configuration dictionary
    config_file = f"{asset_name}.yaml"
    with open(config_file, "w") as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)

    # Define the data asset configuration
    config_data = Data(
        name=asset_name,
        path=config_file,
        version=asset_version,
        type=AssetTypes.URI_FILE,
    )

    # Check if the specified datastore exists, create it if not
    if datastore_name not in workspace.datastores:
        account_name = getpass.getpass(
            prompt="Enter your Azure Storage Account Name: "
        )
        account_key = getpass.getpass(
            prompt="Enter your Azure Storage Account Key: "
        )
        container_name = f"{account_name}-container"
        Datastore.register_azure_blob_container(
            workspace=workspace,
            datastore_name=datastore_name,
            container_name=container_name,
            account_name=account_name,
            account_key=account_key,
        )

    try:
        # Attempt to retrieve the existing asset
        config_asset = ml_client.data.get(
            name=asset_name, version=asset_version
        )
        message = (
            f"Config asset already exists. Name: {config_asset.name}, "
            f"version: {config_asset.version}"
        )
    except Exception:
        # Create a new asset if it does not exist
        config_asset = ml_client.data.create_or_update(config_data)
        message = (
            f"Data asset created. Name: {config_asset.name}, "
            f"version: {config_asset.version}"
        )

    # Print the outcome message
    print(message)

    # Retrieve and return the data asset
    config_asset = ml_client.data.get(name=asset_name, version=asset_version)
    return config_asset


def get_secret(secret_name, key_vault_name):
    """
    Retrieves a secret value from Azure Key Vault.

    Args:
        secret_name (str): The name of the secret.
        key_vault_name (str): The name of the Azure Key Vault.

    Returns:
        str: The retrieved secret value.
    """
    # URL to the Azure Key Vault
    key_vault_url = f"https://{key_vault_name}.vault.azure.net/"

    # Create a credential object using DefaultAzureCredential
    credential = DefaultAzureCredential()

    # Create a SecretClient object for the Key Vault
    client = SecretClient(vault_url=key_vault_url, credential=credential)

    try:
        # Retrieve the secret value
        retrieved_secret = client.get_secret(secret_name)
        return retrieved_secret.value
    except Exception as e:
        print(f"An error occurred accessing the secret: {e}")
        return None


def define_pipeline_components(
    subscription_id,
    resource_group,
    workspace_name,
    pipeline_job_env_name,
    pipeline_job_env_version,
    src_dir,
):
    """
    Define and register the components of the machine learning pipeline.

    Args:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace_name (str): Azure ML workspace name.
        pipeline_job_env_name (str): The name of the environment for the
            pipeline components.
        pipeline_job_env_version (str): The version of the environment for the
            pipeline components.
        src_dir (str): The source directory where the code for the pipeline
            component is located.

    Returns:
        dict: A dictionary of registered pipeline components.
    """

    ml_client = create_ml_client(
        subscription_id, resource_group, workspace_name
    )

    # Define and register the preprocess component of the pipeline
    preprocess_component = command(
        name="preprocess_component",
        display_name="preprocess_component",
        inputs={
            "config": Input(type="uri_file"),
        },
        outputs=dict(
            output_dir=Output(type="uri_folder"),
        ),
        code=src_dir,
        command=(
            "python preprocess.py --config ${{inputs.config}} "
            "--output-dir ${{outputs.output_dir}}"
        ),
        environment=f"{pipeline_job_env_name}:{pipeline_job_env_version}",
    )
    preprocess_component = ml_client.create_or_update(
        preprocess_component.component
    )

    # Define and register the gretel component of the pipeline
    gretel_component = command(
        name="gretel_component",
        display_name="gretel_component",
        inputs={
            "config": Input(type="uri_file"),
            "gretel_api_key": Input(type="string"),
            "input_dir": Input(type="uri_folder"),
        },
        outputs=dict(
            output_dir=Output(type="uri_folder"),
        ),
        code=src_dir,
        command=(
            "python gretel.py --config ${{inputs.config}} "
            "--gretel-api-key ${{inputs.gretel_api_key}} "
            "--input-dir ${{inputs.input_dir}} "
            "--output-dir ${{outputs.output_dir}}"
        ),
        environment=f"{pipeline_job_env_name}:{pipeline_job_env_version}",
    )
    gretel_component = ml_client.create_or_update(gretel_component.component)

    # Define and register the train component of the pipeline
    train_component = command(
        name="train_component",
        display_name="train_component",
        inputs={
            "config": Input(type="uri_file"),
            "input_dir": Input(type="uri_folder"),
            "gretel_dir": Input(type="uri_folder"),
        },
        outputs=dict(
            output_dir=Output(type="uri_folder"),
        ),
        code=src_dir,
        command=(
            "python train.py --config ${{inputs.config}} "
            "--input-dir ${{inputs.input_dir}} "
            "--gretel-dir ${{inputs.gretel_dir}} "
            "--output-dir ${{outputs.output_dir}}"
        ),
        environment=f"{pipeline_job_env_name}:{pipeline_job_env_version}",
    )
    train_component = ml_client.create_or_update(train_component.component)

    # Define and register the evaluate component of the pipeline
    evaluate_component = command(
        name="evaluate_component",
        display_name="evaluate_component",
        inputs={
            "config": Input(type="uri_file"),
            "input_dir": Input(type="uri_folder"),
            "model_dir": Input(type="uri_folder"),
        },
        outputs=dict(
            output_dir=Output(type="uri_folder"),
        ),
        code=src_dir,
        command=(
            "python evaluate.py --config ${{inputs.config}} "
            "--input-dir ${{inputs.input_dir}} "
            "--model-dir ${{inputs.model_dir}} "
            "--output-dir ${{outputs.output_dir}}"
        ),
        environment=f"{pipeline_job_env_name}:{pipeline_job_env_version}",
    )
    evaluate_component = ml_client.create_or_update(
        evaluate_component.component
    )

    # Define and register the register component of the pipeline
    register_component = command(
        name="register_component",
        display_name="register_component",
        inputs={
            "config": Input(type="uri_file"),
            "eval_dir": Input(type="uri_folder"),
            "model_dir": Input(type="uri_folder"),
            "model_display_name": Input(type="string"),
        },
        outputs=dict(
            output_dir=Output(type="uri_folder"),
        ),
        code=src_dir,
        command=(
            "python register.py --config ${{inputs.config}} "
            "--eval-dir ${{inputs.eval_dir}} "
            "--model-dir ${{inputs.model_dir}} "
            "--model-display-name ${{inputs.model_display_name}}"
        ),
        environment=f"{pipeline_job_env_name}:{pipeline_job_env_version}",
    )
    register_component = ml_client.create_or_update(
        register_component.component
    )

    # Return all components
    return {
        "preprocess": preprocess_component,
        "gretel": gretel_component,
        "train": train_component,
        "evaluate": evaluate_component,
        "register": register_component,
    }

# MLOps Pipeline with Gretel Synthetic Data

This repository, located at [gretelai/gretel-mlops](https://github.com/gretelai/gretel-mlops), demonstrates how to leverage Gretel Synthetic data within MLOps pipelines across multiple cloud platforms: AWS SageMaker, Google Cloud Vertex AI, and Azure AI. By integrating synthetic data into machine learning operations, we aim to enhance data privacy, availability, and the robustness of ML models in a world increasingly focused on data security and regulatory compliance.

## Features

- **Notebooks**: Detailed guides to running MLOps pipelines in SageMaker Pipelines, Vertex AI Pipelines, and Azure AI Pipelines, tailored for each cloud provider.
- **Code Repository**: Essential code for constructing these pipelines, providing a practical resource for implementing synthetic data workflows.
- **Config Folder**: A collection of dataset examples ready for ML model training on classification or regression tasks with Gretel Synthetics, showcasing the versatility and effectiveness of synthetic data.

## Supporting Blogpost

Dive deeper into the integration of synthetic data with MLOps by reading our blog post: ["How to Use Amazon SageMaker Pipelines MLOps with Gretel Synthetic Data"](https://aws.amazon.com/blogs/apn/how-to-use-amazon-sagemaker-pipelines-mlops-with-gretel-synthetic-data/). This comprehensive guide illuminates the advantages of synthetic data, detailing integration steps and highlighting the synergy between Gretel and SageMaker Pipelines for privacy-conscious and efficient ML model training.

## Getting Started

### Prerequisites

To begin, ensure you have:

- An account with the relevant cloud provider. Create one at [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/), or [Azure](https://azure.microsoft.com/).   
- Access to Gretel services. Sign up and obtain your API key via the Gretel Console.  

### Installation

1. Clone this repository to your local machine or cloud environment.

```bash
git clone https://github.com/gretelai/gretel-mlops.git
```

We require Python 3.9+ to run Gretel services through the SDK.

## Usage

To utilize MLOps pipelines with Gretel Synthetic data across various cloud platforms, follow these steps. 
Example pipeline configurations that run on public datasets are available in the config folder. 
Each link below directs you to a specific notebook that guides you through the pipeline execution process for the respective cloud service. 
Detailed guidance in each notebook will help you effectively integrate Gretel Synthetic data into your ML workflows.

### Azure AI

For implementing the pipeline in Azure AI, refer to the following notebook:

- [Run Pipeline in Azure AI](https://github.com/gretelai/gretel-mlops/blob/main/run_pipeline_azure.ipynb)


### AWS SageMaker

For deploying and running the pipeline in AWS SageMaker, access the notebook here:

- [Run Pipeline in AWS SageMaker](https://github.com/gretelai/gretel-mlops/blob/main/run_pipeline_sagemaker.ipynb)


### Google Cloud Vertex AI

To use the pipeline with Google Cloud's Vertex AI, follow the instructions in this notebook:

- [Run Pipeline in Google Cloud Vertex AI](https://github.com/gretelai/gretel-mlops/blob/main/run_pipeline_gcp.ipynb)


## Contributing

We welcome contributions to improve this project! Whether you're fixing bugs, adding features, or improving documentation, please let us know how to get involved.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

from setuptools import find_packages, setup

setup(
    name="gretel_mlops",
    version="0.0.2",
    python_requires=">=3.10",
    packages=find_packages("src"),
    package_dir={"": "src"},
    license="https://gretel.ai/license/source-available-license",
    include_package_data=True,
    package_data={
        "gretel_mlops.aws.sagemaker": ["requirements.txt"],
        "gretel_mlops.azure.azureai": ["requirements.yaml"],
    },
)
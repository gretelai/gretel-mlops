from setuptools import find_packages, setup

setup(
    name="gretel_sm_pipelines",
    version="0.0.1",
    python_requires=">=3.9",
    packages=find_packages("src"),
    package_dir={"": "src"},
    license="https://gretel.ai/license/source-available-license",
)
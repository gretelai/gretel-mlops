from setuptools import find_packages, setup

setup(
    name="gretel_sm_pipelines",
    version="0.0.1",
    python_requires=">=3.10",
    packages=find_packages("src"),
    package_dir={"": "src"},
    # install_requires=[
    #     'numpy<1.24,>=1.20',
    #     'wrapt<1.15,>=1.11.0'
    # #     'gretel-client',
    # #     'scikit-learn==1.3.0',
    # #     # 'git+https://github.com/gretelai/gretel-tuner',
    # #     'imblearn',
    # #     'xgboost',
    # #     'optuna'
    # ],
    license="https://gretel.ai/license/source-available-license",
)
import numpy as np

datasets = {
    "abalone": {
        "path": "s3://sagemaker-servicecatalog-seedcode-us-east-1/dataset/abalone-dataset.csv",
        "header": None,
        "feature_columns": {
            "sex": str,
            "length": np.float64,
            "diameter": np.float64,
            "height": np.float64,
            "whole_weight": np.float64,
            "shucked_weight": np.float64,
            "viscera_weight": np.float64,
            "shell_weight": np.float64,
        },
        "label_column": {
            "rings": np.float64
        },
        "label_column_name": "rings",
        "drop_columns": None,
        "objective": "reg:linear",
        "objective_type": "Minimize",
        "ml_eval_metric": "rmse",
        "ml_task": "regression",
        "ml_metric_threshold": 6.0,
    },
    "healthcare-dataset-stroke-data": {
        "path": "s3://gretel-datasets/healthcare-dataset-stroke-data.csv",
        "header": "infer",
        "feature_columns": {
            "id": np.int64,
            "gender": str,
            "age": np.float64,
            "hypertension": np.int64,
            "heart_disease": np.int64,
            "ever_married": str,
            "work_type": str,
            "Residence_type": str,
            "avg_glucose_level": np.float64,
            "bmi": np.float64,
            "smoking_status": str,
        },
        "label_column": {
            "stroke": np.int64,
        },
        "label_column_name": "stroke",
        "drop_columns": [
            "id"
        ],
        "objective": "binary:logistic",
        "objective_type": "Maximize",
        "ml_eval_metric": "auc",
        "ml_task": "classification",
        "ml_metric_threshold": 0.0,
    }
}
    

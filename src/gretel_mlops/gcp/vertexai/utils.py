# import sys
# import subprocess

# subprocess.check_call([
#     sys.executable,
#     "-m", "pip", "install", "--upgrade",
#     "imblearn",
#     "optuna",
#     "numpy",
#     "xgboost"
# ])

import json
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb

# from google.cloud import secretmanager
# from google.api_core.exceptions import GoogleAPICallError
from imblearn.over_sampling import RandomOverSampler
from gretel_client.projects.models import Model
from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix,
        mean_squared_error, mean_absolute_error, r2_score
    )

# def get_gretel_api_key(secret_id, project_number):

#     # Initialize the client
#     client = secretmanager.SecretManagerServiceClient()
#     secret_version = f'projects/{project_number}/secrets/{secret_id}/versions/latest'

#     try:
#         response = client.access_secret_version(request={"name": secret_version})
#         gretel_api_key = response.payload.data.decode("UTF-8")
#         return gretel_api_key

#     except GoogleAPICallError as e:
#         print(f"An error occurred accessing the secret: {e}")
#         return None

def naive_upsample(df, target_column, target_balance=1.0):

    over_sampler = RandomOverSampler(sampling_strategy=target_balance)
    y = df.pop(target_column)
    df_resampled, y_resampled = over_sampler.fit_resample(df, y)
    df_resampled[target_column] = y_resampled

    return df_resampled

def compute_optimal_f1(y_test, predictions):

    # Compute the precision-recall curve

    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    selection = ~((precision==0)&(recall==0))
    precision = precision[selection]
    recall = recall[selection]
    thresholds = thresholds[selection[:-1]]

    # Calculate the f-score
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Find the threshold that maximizes the F1 score
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)

    # Calculate the corresponding precision and recall at the optimal threshold
    optimal_precision = precision[np.argmax(f1_scores)]
    optimal_recall = recall[np.argmax(f1_scores)]

    # Compute the confusion matrix at the optimal threshold
    predictions_binary = (predictions >= best_threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, predictions_binary)

    return best_f1_score, optimal_precision, optimal_recall, conf_matrix


def generate_classification_report(y_test, predictions):
    # Calculate classification metrics
    f1, precision, recall, conf_matrix = compute_optimal_f1(y_test, predictions)

    # Calculate AUC using sklearn's functions
    roc_auc = roc_auc_score(y_test, predictions)
    pr_auc = average_precision_score(y_test, predictions)

    # Create a report dictionary with both the metrics and their standard deviations
    report_dict = {
        "metrics": {
            "auc": {
                "value": roc_auc,
            },
            "aucpr": {
                "value": pr_auc,
            },
            "precision": {
                "value": precision,
            },
            "recall": {
                "value": recall,
            },
            "f1": {
                "value": f1,
            },
            "confusion_matrix": {
                "value": conf_matrix.tolist(),
            }
        },
    }

    return report_dict

def generate_regression_report(y_test, predictions):
    # Calculate regression metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Calculate standard deviations
    mse_std = np.std((y_test - predictions) ** 2)
    rmse_std = np.std(np.abs(y_test - predictions))
    mae_std = np.std(np.abs(y_test - predictions))
    r2_std = np.std(1 - ((y_test - predictions) ** 2) / ((y_test - np.mean(y_test)) ** 2))

    # Create a report dictionary with both the metrics and their standard deviations
    report_dict = {
        "metrics": {
            "mse": {
                "value": mse,
                "std": mse_std
            },
            "mae": {
                "value": mae,
                "std": mae_std
            },
            "R2": {
                "value": r2,
                "std": r2_std
            },
            "rmse": {
                "value": rmse,
                "std": rmse_std
            },
        },
    }

    return report_dict

def objective_func(trial, X_train, y_train, X_val, y_val, task, objective, metric):
    param = {
        'silent': 0,
        'verbosity': 0,
        'objective': objective,
        'eta': trial.suggest_float('eta', 0, 1),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'alpha': trial.suggest_float('alpha', 0, 2),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'num_round': trial.suggest_int('num_round', 100, 500),
        'rate_drop': 0.3,
        'tweedie_variance_power': 1.4
    }

    if task == 'regression':
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = generate_regression_report(y_val, y_pred)['metrics']
    else:
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = generate_classification_report(y_val, y_pred)['metrics']
    return score[metric]['value']


class MLMetric:
    def __init__(self, df_test, preprocess, target_column, metric="f1", task="classification", objective="binary:logistic", objective_type="Maximize"):
        self.df_test = df_test
        self.metric = metric
        self.task = task
        self.preprocess = preprocess
        self.target_column = target_column
        self.objective = objective
        self.objective_type = objective_type

    def __call__(self, model: Model):

        X_train = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
        y_train = X_train.pop(self.target_column)

        # preprocess synthetic data
        X_train = pd.DataFrame(self.preprocess.transform(X_train))

        X_val = self.df_test.copy()
        y_val = X_val.pop(self.target_column)
        X_val.columns = X_train.columns

        study = optuna.create_study(direction=self.objective_type.lower())
        study.optimize(lambda trial: objective_func(trial, X_train, y_train, X_val, y_val, self.task, self.objective, self.metric), n_trials=6, n_jobs=2)

        return study.best_value
        
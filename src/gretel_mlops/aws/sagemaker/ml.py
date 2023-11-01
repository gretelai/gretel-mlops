import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics as ml_metrics
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm

__all__ = ["measure_ml_utility"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
table_format = "grid"

warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

from pdb import set_trace as bp


def naive_upsample(df, target_column="TARGET", target_balance=1.0):
    
    over_sampler = RandomOverSampler(sampling_strategy=target_balance)
    y = df.pop(target_column)
    df_resampled, y_resampled = over_sampler.fit_resample(df, y)
    df_resampled[target_column] = y_resampled
    
    return df_resampled

def compute_optimal_f1(y_test, predictions):

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
        "classification_metrics": {
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
        "regression_metrics": {
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

@dataclass
class MLResults:
    clf: ClassifierMixin
    auc: float
    aucpr: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    hyperparameters: dict
    y_true: np.ndarray
    y_proba: np.ndarray

    def get_scores(self, as_dataframe=False):
        scores = {
            "auc": self.auc,
            "aucpr": self.aucpr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }
        return pd.DataFrame(scores, index=[0]) if as_dataframe else scores

    def print_scores(self):
        print(
            tabulate(
                self.get_scores(as_dataframe=True),
                headers="keys",
                tablefmt=table_format,
                showindex=False,
                floatfmt=".2f",
            )
        )

    def print_optimal_scores(self):
        scores = {}
        for k, v in self.optimal_f1_scores.items():
            scores[f"optimal\n{k}"] = v
        print(
            tabulate(
                pd.DataFrame(scores, index=[0]), headers="keys", tablefmt=table_format, showindex=False, floatfmt=".2f"
            )
        )


def optimize_f1(y_test, y_proba):
    precision, recall, thresholds = ml_metrics.precision_recall_curve(y_test, y_proba)
    f1_array = (2 * precision * recall) / (precision + recall + 1e-8)
    index = np.argmax(f1_array)
    return {
        "f1": f1_array[index],
        "precision": precision[index],
        "recall": recall[index],
        "threshold": thresholds[index],
    }


def measure_ml_utility(
    df_real,
    df_holdout,
    target_column,
    df_boost=None,
    n_splits=3,
    drop_columns=None,
    is_multi_class=False,
    param_grid=None,
    precision_recall_fig_path=None,
):
    df_ref = df_real.copy()

    param_grid = param_grid or {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 7],
        'min_child_weight': [3, 5],
    }

    df_boost_split = None if df_boost is None else [d for d in np.array_split(df_boost.sample(frac=1), n_splits)]

    multi_class = "ovr" if is_multi_class else "raise"
    score_average = "weighted" if is_multi_class else "binary"
    auc_average = "weighted" if is_multi_class else "macro"
    boost_desc = "" if df_boost is None else " with boosted minority"

    param_scores = []
    grid_search = ParameterGrid(param_grid)
    skf = StratifiedKFold(n_splits=n_splits)

    for params in tqdm(grid_search, total=len(grid_search), desc=f"ML utility - Performing grid search{boost_desc}"):
        cv_scores = []
        for split_i, (train_index, test_index) in enumerate(skf.split(df_ref, df_ref[target_column])):
            df_train = df_ref.iloc[train_index]
            df_valid = df_ref.iloc[test_index]

            if df_boost_split is not None:
                df_train = pd.concat([df_train, df_boost_split[split_i]], ignore_index=True)

            y_train = df_train[target_column].astype("category").cat.codes.to_numpy()
            X_train = df_train.drop(columns=[target_column], axis=1)

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train)

            y_valid = df_valid[target_column].astype("category").cat.codes.to_numpy()
            X_valid = df_valid.drop(columns=[target_column], axis=1)
            y_proba = clf.predict_proba(X_valid) if is_multi_class else clf.predict_proba(X_valid)[:, 1]

            cv_scores.append(ml_metrics.roc_auc_score(y_valid, y_proba, multi_class=multi_class, average=auc_average))

        param_scores.append(np.mean(cv_scores))

    best_params = grid_search[np.argmax(param_scores)]
    logger.info(
        "ML utility - Grid search complete -> RF hyperparameters:\n"
        + tabulate(pd.DataFrame(best_params, index=[0]), headers="keys", tablefmt="grid", showindex=False)
        + "\n"
    )

    df_train = df_ref if df_boost is None else pd.concat([df_ref, df_boost], ignore_index=True)
    logger.info(f"ML utility - Final training on {len(df_train)} samples")
    logger.info(f"ML utility - len(real) = {len(df_ref)}, len(boost) = {len([] if df_boost is None else df_boost)}")

    y_train = df_train[target_column].astype("category").cat.codes.to_numpy()
    X_train = df_train.drop(columns=[target_column], axis=1)

    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X_train, y_train)

    y_test = df_holdout[target_column].astype("category").cat.codes.to_numpy()
    X_test = df_holdout.drop(columns=[target_column], axis=1)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if is_multi_class else clf.predict_proba(X_test)[:, 1]
    
    logger.info(
        "ML utility - Classification Report:\n"
        f"{ml_metrics.classification_report(y_test, y_pred, target_names=['negative', 'positive'])}"
    )

    if is_multi_class:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        n_classes = y_test_bin.shape[1]
        auc_scores = []
        for i in range(n_classes):
            precision, recall, _ = ml_metrics.precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
            auc_scores.append(ml_metrics.auc(recall, precision))
        aucpr_score = np.mean(auc_scores)
    else:
        precision, recall, _ = ml_metrics.precision_recall_curve(y_test, y_proba)
        aucpr_score = ml_metrics.auc(recall, precision)

    optimized_metrics = optimize_f1(y_test, y_proba)
    f1 = optimized_metrics["f1"]
    precision = optimized_metrics["precision"]
    recall = optimized_metrics["recall"]
    
    return MLResults(
        clf=clf,
        auc=ml_metrics.roc_auc_score(y_test, y_proba, multi_class=multi_class, average=auc_average),
        aucpr=aucpr_score,
        f1=f1,
        precision=precision,
        recall=recall,
        accuracy=ml_metrics.accuracy_score(y_test, y_pred),
        hyperparameters=best_params,
        y_true=y_test,
        y_proba=y_proba,
    )




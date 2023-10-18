"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import pickle
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from datasets import datasets
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name
    dataset_path = datasets[dataset_name]['path']
    dataset_header = datasets[dataset_name]['header']
    feature_columns = datasets[dataset_name]['feature_columns']
    feature_columns_names = list(feature_columns.keys())
    label_column = datasets[dataset_name]['label_column']
    label_column_name = datasets[dataset_name]['label_column_name']
    drop_columns = datasets[dataset_name]['drop_columns']
    categorical_features = [key for key, value in feature_columns.items() if value == str]

    bucket = dataset_path.split("/")[2]
    key = "/".join(dataset_path.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    # For headerless CSV file we need to specify the column names.
    if dataset_header:
        df = pd.read_csv(
            fn,
            dtype=merge_two_dicts(feature_columns, label_column),
        )
    else:
        df = pd.read_csv(
            fn,
            header=None,
            names=feature_columns_names + [label_column_name],
            dtype=merge_two_dicts(feature_columns, label_column),
        )
    os.unlink(fn)
    
    if drop_columns:
        df.drop(drop_columns, axis=1, inplace=True)
        used_cols = [col for col in list(feature_columns_names) if col not in drop_columns]
    else:
        used_cols = list(feature_columns_names)
        
    logger.debug("Defining transformers.")
    numeric_features = [col for col in used_cols if col not in categorical_features]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Applying transforms.")
    X = df.sample(frac=1).reset_index(drop=True)
    y = X.pop(label_column_name)
    preprocess.fit(X)
    
    logger.info("Writing out preprocessing object to %s.", base_dir)
    preprocess_path = f"{base_dir}/preprocess/preprocess.pkl"
    with open(preprocess_path, 'wb') as file:
        pickle.dump(preprocess, file)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    logger.info("Writing out datasets to %s.", base_dir)
    # the first variable is assumed to be the target variable
    train_pre = pd.DataFrame(preprocess.transform(X_train))
    train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1) 
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    
    validation_pre = pd.DataFrame(preprocess.transform(X_valid))
    validation = pd.concat([y_valid.reset_index(drop=True), validation_pre], axis=1)
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    
    test_pre = pd.DataFrame(preprocess.transform(X_test))
    test = pd.concat([y_test.reset_index(drop=True), test_pre], axis=1)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    train = X_train
    train[label_column_name] = y_train
    train.to_csv(f"{base_dir}/train_source/train.csv", header=True, index=False)
    
    
    


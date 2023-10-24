"""Feature engineers the abalone dataset."""
import argparse
import logging
import json
import os
import pathlib
import pickle
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from pdb import set_trace as bp


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


def read_data(path, base_dir):
    bucket = path.split("/")[2]
    key = "/".join(path.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/train.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)
    
    return df
    


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--validation-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--ml-task", type=str, required=True)
    parser.add_argument("--drop-columns", type=str, nargs='+', default=[], help="List of column names to drop")
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    # base_dir = "/home/sagemaker-user/tmp1"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    train_path = args.train_path
    validation_path = args.validation_path
    test_path = args.test_path
    ml_task = args.ml_task
    target_column = args.target_column
    drop_columns = args.drop_columns
    
    df = read_data(train_path, base_dir)       
    feature_columns = [col for col in df.columns if col != target_column]
    if drop_columns:
        df.drop(drop_columns, axis=1, inplace=True)
        used_cols = [col for col in feature_columns if col not in drop_columns]
    else:
        used_cols = feature_columns
    
    logger.debug("Defining transformers.")
    categorical_features = df[used_cols].select_dtypes(include=['object', 'category']).columns.tolist()
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
        ],
        sparse_threshold=0
    )

    logger.info("Convert target colum into integer categories.")
    if ml_task == "classification":
        df[target_column] = pd.Categorical(df[target_column])
        df[target_column] = df[target_column].cat.codes
    
    logger.info("Applying transforms.")
    X_train = df.sample(frac=1).reset_index(drop=True)
    y_train = X_train.pop(target_column)
    preprocess.fit(X_train)
    
    logger.info("Writing out preprocessing object to %s.", base_dir)
    preprocess_path = f"{base_dir}/preprocess/preprocess.pkl"
    # preprocess_path = f"tmp/preprocess.pkl"
    with open(preprocess_path, 'wb') as file:
        pickle.dump(preprocess, file)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X_train))
    
    if test_path:
        df_test = read_data(test_path, base_dir)
        X_test = df_test.sample(frac=1).reset_index(drop=True)
        y_test = X_test.pop(target_column)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, 
            test_size=0.20, 
            random_state=42, 
            stratify=y_train if ml_task == "classification" else None
        )
    
    if validation_path:
        df_valid = read_data(validation_path, base_dir)
        X_valid = df_valid.sample(frac=1).reset_index(drop=True)
        y_valid = X_valid.pop(target_column)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, 
            test_size=0.25, 
            random_state=42, 
            stratify=y_train if ml_task == "classification" else None
        )

    logger.info("Writing out datasets to %s.", base_dir)
    # the first variable is assumed to be the target variable
    train_pre = pd.DataFrame(preprocess.transform(X_train))
    train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1) 
    train.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    
    validation_pre = pd.DataFrame(preprocess.transform(X_valid))
    validation = pd.concat([y_valid.reset_index(drop=True), validation_pre], axis=1)
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=True, index=False
    )
    
    test_pre = pd.DataFrame(preprocess.transform(X_test))
    test = pd.concat([y_test.reset_index(drop=True), test_pre], axis=1)
    test.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    
    train = X_train
    train[target_column] = y_train
    train.to_csv(f"{base_dir}/train_source/train.csv", header=True, index=False)
    
    
    


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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64,
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/abalone-dataset.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    os.unlink(fn)

    logger.debug("Defining transformers.")
    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["sex"]
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
    y = X.pop("rings")
    preprocess.fit(X)
    
    logger.info("Writing out preprocessing object to %s.", base_dir)
    preprocess_path = f"{base_dir}/preprocess/preprocess.pkl"
    with open(preprocess_path, 'wb') as file:
        pickle.dump(preprocess, file)

    # X_pre = preprocess.fit_transform(df)
    # y_pre = y.to_numpy().reshape(len(y), 1)
    # X = np.concatenate((y_pre, df), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # np.random.shuffle(X)
    # train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)

    train_pre = pd.DataFrame(preprocess.transform(X_train))
    train = pd.concat([train_pre, y_train.reset_index(drop=True)], axis=1)
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    
    validation_pre = pd.DataFrame(preprocess.transform(X_valid))
    validation = pd.concat([validation_pre, y_valid.reset_index(drop=True)], axis=1)
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    
    test_pre = pd.DataFrame(preprocess.transform(X_test))
    test = pd.concat([test_pre, y_test.reset_index(drop=True)], axis=1)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    train = X_train
    train['rings'] = y_train
    train.to_csv(f"{base_dir}/train_source/train.csv", header=True, index=False)
    
    
    


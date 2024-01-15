"""Script to preprocess source data."""

import argparse
import logging
import os
import pathlib
import joblib

import boto3
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def read_data(path, base_dir):
    """
    Reads data from a given S3 path and downloads it to a specified directory.

    Args:
        path (str): S3 path to the dataset.
        base_dir (str): Local directory path to download the dataset.

    Returns:
        DataFrame: A pandas DataFrame containing the dataset.
    """
    bucket = path.split("/")[2]
    key = "/".join(path.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/train.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)

    return df


if __name__ == "__main__":
    # Setup argparse for command line arguments
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    # Define the command line arguments
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--validation-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--ml-task", type=str, required=True)
    parser.add_argument(
        "--drop-columns",
        type=str,
        nargs="+",
        default=[],
        help="List of column names to drop",
    )
    args = parser.parse_args()

    # Define base directory for processing
    base_dir = "/opt/ml/processing"
    # Create necessary directories
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    # Extract arguments
    train_path = args.train_path
    validation_path = args.validation_path
    test_path = args.test_path
    ml_task = args.ml_task
    target_column = args.target_column
    drop_columns = args.drop_columns

    # Read and preprocess training data
    logger.info("Reading in dataset.")
    df = read_data(train_path, base_dir)
    # Define feature columns
    feature_columns = [col for col in df.columns if col != target_column]
    # Drop specified columns if any
    if drop_columns:
        df.drop(drop_columns, axis=1, inplace=True)
        used_cols = [col for col in feature_columns if col not in drop_columns]
    else:
        used_cols = feature_columns

    # Setup transformers for numeric and categorical features
    logger.info("Defining transformers.")
    categorical_features = (
        df[used_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )
    numeric_features = [
        col for col in used_cols if col not in categorical_features
    ]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
    )

    # Convert target column for classification tasks
    logger.info("Convert target column into integer categories.")
    if ml_task == "classification":
        df[target_column] = pd.Categorical(df[target_column])
        df[target_column] = df[target_column].cat.codes

    # Apply transforms to the data
    logger.info("Applying transforms.")
    X_train = df.sample(frac=1).reset_index(drop=True)
    y_train = X_train.pop(target_column)
    preprocess.fit(X_train)

    # Serialize and save the preprocessing object
    logger.info("Writing out preprocessing object to %s.", base_dir)
    preprocess_path = f"{base_dir}/preprocess/preprocess.pkl"
    joblib.dump(preprocess, preprocess_path)

    # Split data into train, validation, and test datasets
    logger.info(
        "Splitting %d rows of data into train, validation, test datasets.",
        len(X_train),
    )

    # Check if a separate test dataset path is provided
    if test_path:
        logger.info("Processing test dataset.")
        # Read and preprocess test data
        df_test = read_data(test_path, base_dir)
        X_test = df_test.sample(frac=1).reset_index(drop=True)
        y_test = X_test.pop(target_column)
    else:
        # Split the training data into training and test sets
        logger.info("Splitting train dataset into train and test subsets.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=42,
            stratify=y_train if ml_task == "classification" else None,
        )

    # Check if a separate validation dataset path is provided
    if validation_path:
        logger.info("Processing validation dataset.")
        # Read and preprocess validation data
        df_valid = read_data(validation_path, base_dir)
        X_valid = df_valid.sample(frac=1).reset_index(drop=True)
        y_valid = X_valid.pop(target_column)
    else:
        # Split the training data into training and validation sets
        logger.info(
            "Splitting train dataset into train and validation subsets."
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=42,
            stratify=y_train if ml_task == "classification" else None,
        )

    # Process and save the train dataset
    logger.info("Writing out train dataset to %s.", base_dir)
    train_pre = pd.DataFrame(preprocess.transform(X_train))
    train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1)
    train.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)

    # Process and save the validation dataset
    logger.info("Writing out validation dataset to %s.", base_dir)
    validation_pre = pd.DataFrame(preprocess.transform(X_valid))
    validation = pd.concat(
        [y_valid.reset_index(drop=True), validation_pre], axis=1
    )
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=True, index=False
    )

    # Process and save the test dataset
    logger.info("Writing out test dataset to %s.", base_dir)
    test_pre = pd.DataFrame(preprocess.transform(X_test))
    test = pd.concat([y_test.reset_index(drop=True), test_pre], axis=1)
    test.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)

    # Save the original training data for further reference or use
    logger.info("Writing out original training data.")
    train = X_train
    train[target_column] = y_train
    train.to_csv(
        f"{base_dir}/train_source/train.csv", header=True, index=False
    )

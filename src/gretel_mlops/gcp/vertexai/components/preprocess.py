# Preprocess step

from kfp.dsl import component, OutputPath

@component(
    base_image="python:3.10",
    packages_to_install=['scikit-learn', 'gcsfs', 'pandas', 'google-cloud-aiplatform'],
)
def preprocess_component(
    config: str,
    output_dir: OutputPath()
):

  import argparse
  import logging
  import json
  import os

  import pandas as pd

  from google.cloud import storage
  from sklearn.compose import ColumnTransformer
  from sklearn.impute import SimpleImputer
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.model_selection import train_test_split
  from joblib import dump

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler())

  # parse config file
  config = json.loads(config)
  train_path = config['dataset']['train_path']
  test_path = config['dataset']['test_path']
  validation_path = config['dataset']['validation_path']
  target_column = config['dataset']['target_column']
  drop_columns = config['dataset']['drop_columns']
  ml_task = config['ML']['ml_task']

  # Ensure the output directory exists
  os.makedirs(output_dir, exist_ok=True)

  logger.info("Reading in dataset.")
  df = pd.read_csv(train_path)
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

  logger.info("Writing out preprocessing object")
  dump(preprocess, f"{output_dir}/preprocess.joblib")

  logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X_train))
  if test_path:
      df_test = pd.read_csv(test_path)
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
      df_valid = pd.read_csv(validation_path)
      X_valid = df_valid.sample(frac=1).reset_index(drop=True)
      y_valid = X_valid.pop(target_column)
  else:
      X_train, X_valid, y_train, y_valid = train_test_split(
          X_train, y_train,
          test_size=0.25,
          random_state=42,
          stratify=y_train if ml_task == "classification" else None
      )

  logger.info("Writing out datasets")
  # the first variable is assumed to be the target variable
  train_pre = pd.DataFrame(preprocess.transform(X_train))
  train = pd.concat([y_train.reset_index(drop=True), train_pre], axis=1)
  train.to_csv(f"{output_dir}/train.csv", header=True, index=False)

  validation_pre = pd.DataFrame(preprocess.transform(X_valid))
  validation = pd.concat([y_valid.reset_index(drop=True), validation_pre], axis=1)
  validation.to_csv(f"{output_dir}/validation.csv", header=True, index=False)

  test_pre = pd.DataFrame(preprocess.transform(X_test))
  test = pd.concat([y_test.reset_index(drop=True), test_pre], axis=1)
  test.to_csv(f"{output_dir}/test.csv", header=True, index=False)

  train_source = X_train
  train_source[target_column] = y_train
  train_source.to_csv(f"{output_dir}/train_source.csv", header=True, index=False)

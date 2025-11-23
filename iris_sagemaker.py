from __future__ import print_function

import argparse
import os
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def model_fn(model_dir):
    """
    SageMaker loads the model for inference by calling this.
    We just load the sklearn Pipeline that we saved during training.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    # ========= 0. Parse SageMaker environment arguments =========
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )

    args = parser.parse_args()

    #Load data
    data_path = os.path.join(args.train, "Iris.csv")
    iris = pd.read_csv(data_path)

    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    target_col = "Species"

    X = iris[feature_cols]
    y = iris[target_col]

    #Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=14, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=14, stratify=y_temp
    )

    print("Shapes -> train: {}, val: {}, test: {}".format(
        X_train.shape, X_val.shape, X_test.shape
    ))

    #Build Pipeline: MinMaxScaler + LogisticRegression
    pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )

    #Train
    pipeline.fit(X_train, y_train)

 

    #Save the trained pipeline
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nPipeline saved to: {model_path}")

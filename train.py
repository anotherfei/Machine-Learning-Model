"""
Train ONE model end to end: preprocess -> feature engineer -> fit -> evaluate -> save.

Usage:
    python train.py --model lightgbm
    python train.py --model random_forest
"""

import argparse
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import preprocessing
import feature_engineering
import metrics
import artifact_utils
from models import MODEL_REGISTRY


def get_or_build_features() -> pd.DataFrame:
    if not os.path.exists(config.FEATURES_DATA_PATH):
        print("[train] No cached features found — running full preprocessing pipeline.")
        preprocessing.run_preprocessing()
        return feature_engineering.run_feature_engineering()

    print(f"[train] Loading cached features from {config.FEATURES_DATA_PATH}")
    return pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])


def main(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(MODEL_REGISTRY.keys())}")

    df = get_or_build_features()
    train_df, test_df = preprocessing.split_data(df)

    feature_cols = feature_engineering.get_feature_columns(df)
    X_train, y_train = train_df[feature_cols], train_df[config.COL_RUL]
    X_test, y_test = test_df[feature_cols], test_df[config.COL_RUL]

    wrapper = MODEL_REGISTRY[model_name]()

    scaler = None
    if wrapper.needs_scaling:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    wrapper.fit(X_train, y_train)
    y_pred = wrapper.predict(X_test)

    scores = metrics.evaluate(y_test, y_pred)
    metrics.print_scores(model_name, scores)

    artifact_utils.save_model_artifacts(model_name, wrapper.model, feature_cols, scaler)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()
    main(args.model)

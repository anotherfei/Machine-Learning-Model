"""
Inference on new sensor data using a saved model. Fails loudly on any
mismatch between what the model expects and what feature_engineering.py
currently produces — a wrong-but-silent prediction is worse than a crash.

Usage:
    python predict.py --model lightgbm --input new_sensor_data.csv
"""

import argparse
import pandas as pd

import config
import preprocessing
import feature_engineering
import artifact_utils


def validate_and_align_features(X: pd.DataFrame, expected: list) -> pd.DataFrame:
    actual = list(X.columns)
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)

    if missing:
        raise ValueError(f"Missing expected features: {sorted(missing)}")
    if extra:
        raise ValueError(
            f"Unexpected extra features: {sorted(extra)}. "
            f"This usually means feature_engineering.py was changed since the "
            f"model was trained without retraining."
        )

    # Reorder to match training order — column order matters even when names match
    return X[expected]


def predict(model_name: str, input_csv: str) -> pd.DataFrame:
    model, feature_cols, scaler, meta = artifact_utils.load_model_artifacts(model_name)

    raw_df = pd.read_csv(input_csv, parse_dates=[config.COL_TIMESTAMP])
    clean_df = preprocessing.clean_data(raw_df)
    feat_df = feature_engineering.create_features(clean_df)

    if feat_df.empty:
        raise ValueError(
            "No rows survived feature engineering — likely not enough history "
            f"per unit to fill a window of size {config.WINDOW_SIZE} "
            f"(config.WINDOW_SIZE). Provide more historical rows per unit."
        )

    candidate_cols = feature_engineering.get_feature_columns(feat_df)
    X = validate_and_align_features(feat_df[candidate_cols], feature_cols)

    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    predictions = model.predict(X)

    result = feat_df[[config.COL_UNIT_ID, config.COL_TIMESTAMP]].copy()
    result["predicted_RUL"] = predictions
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="Path to new sensor data CSV")
    args = parser.parse_args()

    output = predict(args.model, args.input)
    print(output.to_string(index=False))

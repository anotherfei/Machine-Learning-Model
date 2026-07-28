"""
Train ALL models under identical GroupKFold splits (grouped by unit_id) and
print a comparison table. This is the fair-comparison entry point —
train.py trains one model on one holdout split; this script cross-validates
every model on the same folds so results are directly comparable.

Usage:
    python compare_models.py
"""

import os
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import config
import preprocessing
import feature_engineering
import metrics
from models import MODEL_REGISTRY


def get_or_build_features() -> pd.DataFrame:
    if not os.path.exists(config.FEATURES_DATA_PATH):
        preprocessing.run_preprocessing()
        return feature_engineering.run_feature_engineering()
    return pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])


def run_cv_for_model(model_cls, X, y, groups, feature_cols):
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        wrapper = model_cls()

        if wrapper.needs_scaling:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

        wrapper.fit(X_train, y_train)
        y_pred = wrapper.predict(X_test)

        scores = metrics.evaluate(y_test, y_pred)
        scores["fold"] = fold
        fold_scores.append(scores)

    return pd.DataFrame(fold_scores)


def main():
    df = get_or_build_features()
    feature_cols = feature_engineering.get_feature_columns(df)

    X = df[feature_cols]
    y = df[config.COL_RUL]
    groups = df[config.COL_UNIT_ID]

    n_units = groups.nunique()
    if n_units < config.N_SPLITS:
        raise ValueError(
            f"Only {n_units} unique units but N_SPLITS={config.N_SPLITS} in config.py. "
            f"GroupKFold needs at least as many groups as folds — lower N_SPLITS "
            f"or add more units."
        )

    results = {}
    for name, model_cls in MODEL_REGISTRY.items():
        print(f"\n[compare_models] Running {name} ({config.N_SPLITS}-fold GroupKFold)...")
        fold_df = run_cv_for_model(model_cls, X, y, groups, feature_cols)
        results[name] = fold_df

        mean_scores = fold_df.drop(columns="fold").mean()
        std_scores = fold_df.drop(columns="fold").std()
        summary = " | ".join(f"{k}: {mean_scores[k]:.3f} +/- {std_scores[k]:.3f}" for k in mean_scores.index)
        print(f"[{name}] {summary}")

    # Final comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (mean across folds)")
    print("=" * 80)
    summary_rows = []
    for name, fold_df in results.items():
        row = fold_df.drop(columns="fold").mean().to_dict()
        row["model"] = name
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_df = summary_df[["MAE", "RMSE", "R2", "PHM08"]].sort_values("PHM08")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    best_model = summary_df.index[0]
    print(f"\nBest model by PHM08 score: {best_model}")
    print("Note: PHM08 ranking may differ from RMSE ranking — PHM08 penalizes "
          "late (optimistic) predictions harder, which matters more operationally.")

    return summary_df


if __name__ == "__main__":
    main()

"""
Train classical regression models using the v8 selected feature dataset.

This script reads the paper-friendly v8 feature table, splits it by the saved
chronological train/test label, trains Linear Regression, Ridge Regression, and
Support Vector Regression, and saves evaluation outputs for reporting.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR


YEAR_COL = "Year"
SPLIT_COL = "dataset_split"
TARGET_COL = "yield_kg_ha"


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def resolve_default_paths():
    """Resolve default project input and output paths."""
    training_dir = os.path.dirname(os.path.abspath(__file__))
    classical_dir = os.path.dirname(training_dir)
    project_dir = os.path.dirname(classical_dir)
    feature_path = os.path.join(
        project_dir,
        "artifacts",
        "classical",
        "features",
        "Feature_result_v8",
        "barley_features_selected_scaled_v8.csv",
    )
    output_dir = os.path.join(project_dir, "artifacts", "classical", "training")
    return feature_path, output_dir


def parse_args():
    """Parse reusable command-line paths."""
    default_features, default_output = resolve_default_paths()
    parser = argparse.ArgumentParser(
        description="Train classical regression models on engineered barley features."
    )
    parser.add_argument("--features", default=default_features)
    parser.add_argument("--output-dir", default=default_output)
    return parser.parse_args()


def load_feature_data(feature_path):
    """Load v8 selected feature data and prepare train/test arrays."""
    df = pd.read_csv(feature_path)
    required_cols = {YEAR_COL, SPLIT_COL, TARGET_COL}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    feature_cols = [
        col
        for col in df.columns
        if col not in [YEAR_COL, SPLIT_COL, TARGET_COL]
    ]

    train_df = df[df[SPLIT_COL] == "train"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("The v8 feature table must contain train and test rows.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    return df, feature_cols, train_df, test_df, X_train, y_train, X_test, y_test


def build_models():
    """Create the classical model set."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector Regression": SVR(
            kernel="linear",
            C=500.0,
            epsilon=1.0,
        ),
    }


def train_models(models, X_train, y_train):
    """Fit all models on the training data."""
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def save_results(
    models,
    feature_cols,
    train_df,
    test_df,
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir,
):
    """Save metrics, predictions, model coefficients, and metadata."""
    metrics_rows = []
    predictions = test_df[[YEAR_COL, TARGET_COL]].rename(
        columns={TARGET_COL: "actual_yield_kg_ha"}
    )
    train_predictions = train_df[[YEAR_COL, TARGET_COL]].rename(
        columns={TARGET_COL: "actual_yield_kg_ha"}
    )

    for model_name, model in models.items():
        test_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        metrics = calculate_metrics(y_test, test_pred)

        metrics_rows.append(
            {
                "model": model_name,
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
            }
        )

        prefix = model_name.lower().replace(" ", "_")
        predictions[f"{prefix}_prediction"] = test_pred
        predictions[f"{prefix}_residual"] = predictions["actual_yield_kg_ha"] - test_pred

        train_predictions[f"{prefix}_prediction"] = train_pred
        train_predictions[f"{prefix}_residual"] = (
            train_predictions["actual_yield_kg_ha"] - train_pred
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE")
    metrics_df.to_csv(
        os.path.join(output_dir, "metrics_v3.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        os.path.join(output_dir, "predictions_v3.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    train_predictions.to_csv(
        os.path.join(output_dir, "train_predictions_v3.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    coefficient_rows = []
    for model_name, model in models.items():
        if hasattr(model, "coef_"):
            for feature, coefficient in zip(feature_cols, model.coef_):
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "coefficient": coefficient,
                    }
                )

    if coefficient_rows:
        pd.DataFrame(coefficient_rows).to_csv(
            os.path.join(output_dir, "linear_coefficients_v3.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    metadata = {
        "feature_source": "Feature_result_v8/barley_features_selected_scaled_v8.csv",
        "target": TARGET_COL,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "train_years": [int(train_df[YEAR_COL].min()), int(train_df[YEAR_COL].max())],
        "test_years": [int(test_df[YEAR_COL].min()), int(test_df[YEAR_COL].max())],
        "models": list(models.keys()),
        "svr_parameters": {
            "kernel": "linear",
            "C": 500.0,
            "epsilon": 1.0,
            "setting": "linear_high_c",
        },
        "prediction_setting": "next-year yield prediction using lagged historical features",
    }
    with open(
        os.path.join(output_dir, "training_metadata_v3.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    return metrics_df


def main():
    args = parse_args()
    feature_path = args.features
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    (
        _,
        feature_cols,
        train_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = load_feature_data(feature_path)

    models = train_models(build_models(), X_train, y_train)
    metrics_df = save_results(
        models=models,
        feature_cols=feature_cols,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
    )

    print("Training v3 completed.")
    print(f"Feature source: {feature_path}")
    print(f"Output folder: {output_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

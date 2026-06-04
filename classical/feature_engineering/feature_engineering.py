"""
Feature engineering for next-year barley yield prediction.

This script reads the cleaned annual agricultural dataset, creates
leakage-safe historical features, performs time-based train/test splitting,
selects features using the training set only, applies StandardScaler using the
training set only, and saves reusable outputs for model training.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BarleyNextYearFeatureEngineering:
    """Build leakage-safe features for annual next-year yield prediction."""

    def __init__(
        self,
        year_col="Year",
        target_col="yield_kg_ha",
        temp_col="ave_temp",
        precip_col="Precipitation_mm",
        output_col="output_ha",
        train_ratio=0.8,
        correlation_threshold=0.90,
    ):
        self.year_col = year_col
        self.target_col = target_col
        self.temp_col = temp_col
        self.precip_col = precip_col
        self.output_col = output_col
        self.train_ratio = train_ratio
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()

        self.candidate_features = [
            "time_index",
            "time_index_squared",
            "yield_lag1",
            "yield_lag2",
            "yield_ma3_shifted",
            "temp_lag1",
            "precip_lag1",
            "precip_ma3_shifted",
            "drought_index_lag1",
            "temp_optimal_distance_lag1",
            "precip_optimal_distance_lag1",
            "yield_pct_change_lag1",
        ]
        self.priority_features = [
            "time_index",
            "yield_lag1",
            "yield_lag2",
            "temp_lag1",
            "precip_lag1",
            "yield_ma3_shifted",
            "precip_ma3_shifted",
            "drought_index_lag1",
        ]
        self.selected_features = []
        self.removed_features = []

    def load_cleaned_data(self, file_path):
        """Load cleaned data and sort by year."""
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Input file must be a CSV or Excel file.")

        required_cols = [
            self.year_col,
            self.target_col,
            self.temp_col,
            self.precip_col,
            self.output_col,
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df.sort_values(self.year_col).reset_index(drop=True)
        print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Year range: {df[self.year_col].min()}-{df[self.year_col].max()}")
        return df

    def construct_features(self, df):
        """Create features that only use information available before year t."""
        df = df.copy()

        df["time_index"] = np.arange(len(df))
        df["time_index_squared"] = df["time_index"] ** 2

        df["yield_lag1"] = df[self.target_col].shift(1)
        df["yield_lag2"] = df[self.target_col].shift(2)
        df["yield_ma3_shifted"] = (
            df[self.target_col].shift(1).rolling(window=3).mean()
        )

        df["temp_lag1"] = df[self.temp_col].shift(1)
        df["precip_lag1"] = df[self.precip_col].shift(1)
        df["precip_ma3_shifted"] = (
            df[self.precip_col].shift(1).rolling(window=3).mean()
        )

        df["drought_index_lag1"] = (
            df[self.precip_col].shift(1) / (df[self.temp_col].shift(1) * 20)
        )
        df["temp_optimal_distance_lag1"] = np.abs(df[self.temp_col].shift(1) - 20)
        df["precip_optimal_distance_lag1"] = np.abs(
            df[self.precip_col].shift(1) - 500
        )
        df["yield_pct_change_lag1"] = df[self.target_col].pct_change().shift(1)

        output_cols = [self.year_col, self.target_col] + self.candidate_features
        df = df[output_cols]

        before_drop = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=self.candidate_features + [self.target_col])
        df = df.reset_index(drop=True)

        print(f"Constructed candidate features: {len(self.candidate_features)}")
        print(f"Dropped rows after lag and rolling features: {before_drop - len(df)}")
        print(f"Remaining rows: {len(df)}")
        return df

    def split_by_time(self, df):
        """Use earliest rows as training data and latest rows as test data."""
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        split_idx = int(len(df) * self.train_ratio)
        if split_idx == 0 or split_idx == len(df):
            raise ValueError("Train/test split produced an empty subset.")

        df = df.copy()
        df["dataset_split"] = "train"
        df.loc[df.index >= split_idx, "dataset_split"] = "test"

        train_df = df[df["dataset_split"] == "train"].copy()
        test_df = df[df["dataset_split"] == "test"].copy()

        print(f"Training rows: {len(train_df)}")
        print(f"Test rows: {len(test_df)}")
        print(f"Training period: {train_df[self.year_col].min()}-{train_df[self.year_col].max()}")
        print(f"Test period: {test_df[self.year_col].min()}-{test_df[self.year_col].max()}")
        return df, train_df, test_df

    def remove_invalid_features(self, train_df):
        """Remove features that are missing, infinite, or constant in training data."""
        valid_features = []
        invalid_features = []

        for feature in self.candidate_features:
            series = train_df[feature].replace([np.inf, -np.inf], np.nan)
            if series.isna().all():
                invalid_features.append((feature, "fully_missing"))
            elif series.nunique(dropna=True) <= 1:
                invalid_features.append((feature, "constant"))
            else:
                valid_features.append(feature)

        return valid_features, invalid_features

    def choose_feature_to_remove(self, feature_a, feature_b):
        """Choose which feature to remove from a highly correlated pair."""
        a_priority = feature_a in self.priority_features
        b_priority = feature_b in self.priority_features

        if a_priority and not b_priority:
            return feature_b
        if b_priority and not a_priority:
            return feature_a

        a_rank = (
            self.priority_features.index(feature_a)
            if a_priority
            else len(self.priority_features)
        )
        b_rank = (
            self.priority_features.index(feature_b)
            if b_priority
            else len(self.priority_features)
        )
        if a_rank < b_rank:
            return feature_b
        if b_rank < a_rank:
            return feature_a

        return feature_b

    def select_features(self, train_df):
        """Remove highly correlated features using training data only."""
        valid_features, invalid_features = self.remove_invalid_features(train_df)
        selected = list(valid_features)
        removed = [
            {"feature": feature, "reason": reason, "correlated_with": ""}
            for feature, reason in invalid_features
        ]

        corr_matrix = train_df[valid_features].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_pairs = []
        for feature_a in upper_triangle.index:
            for feature_b in upper_triangle.columns:
                corr_value = upper_triangle.loc[feature_a, feature_b]
                if pd.notna(corr_value) and corr_value > self.correlation_threshold:
                    high_corr_pairs.append((feature_a, feature_b, corr_value))

        high_corr_pairs = sorted(high_corr_pairs, key=lambda item: item[2], reverse=True)

        for feature_a, feature_b, corr_value in high_corr_pairs:
            if feature_a not in selected or feature_b not in selected:
                continue

            feature_to_remove = self.choose_feature_to_remove(feature_a, feature_b)
            feature_to_keep = feature_b if feature_to_remove == feature_a else feature_a
            selected.remove(feature_to_remove)
            removed.append(
                {
                    "feature": feature_to_remove,
                    "reason": f"high_correlation_{corr_value:.3f}",
                    "correlated_with": feature_to_keep,
                }
            )

        self.selected_features = selected
        self.removed_features = removed

        print(f"Valid candidate features: {len(valid_features)}")
        print(f"Selected features: {len(selected)}")
        print(f"Removed features: {len(removed)}")
        return selected, removed, corr_matrix

    def scale_train_test(self, train_df, test_df, selected_features):
        """Fit StandardScaler on training data and transform train and test data."""
        X_train = train_df[selected_features].copy()
        X_test = test_df[selected_features].copy()
        y_train = train_df[[self.target_col]].copy()
        y_test = test_df[[self.target_col]].copy()
        train_years = train_df[[self.year_col]].copy()
        test_years = test_df[[self.year_col]].copy()

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=selected_features,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=selected_features,
            index=X_test.index,
        )

        print("Scaled train and test features with StandardScaler")
        return X_train_scaled, X_test_scaled, y_train, y_test, train_years, test_years

    def save_outputs(
        self,
        engineered_df,
        selected_features,
        removed_features,
        corr_matrix,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        train_years,
        test_years,
        output_dir,
    ):
        """Save reusable feature engineering outputs."""
        feature_result_dir = os.path.join(output_dir, "Feature_result_v8")
        os.makedirs(feature_result_dir, exist_ok=True)

        engineered_df.to_csv(
            os.path.join(feature_result_dir, "barley_engineered_unscaled_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.Series(selected_features, name="feature").to_csv(
            os.path.join(feature_result_dir, "selected_features_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(removed_features).to_csv(
            os.path.join(feature_result_dir, "removed_features_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        corr_matrix.to_csv(
            os.path.join(feature_result_dir, "training_feature_correlation_v8.csv"),
            encoding="utf-8-sig",
        )

        X_train_scaled.to_csv(
            os.path.join(feature_result_dir, "X_train_scaled_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        X_test_scaled.to_csv(
            os.path.join(feature_result_dir, "X_test_scaled_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        y_train.to_csv(
            os.path.join(feature_result_dir, "y_train_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        y_test.to_csv(
            os.path.join(feature_result_dir, "y_test_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        train_years.to_csv(
            os.path.join(feature_result_dir, "train_years_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        test_years.to_csv(
            os.path.join(feature_result_dir, "test_years_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        train_scaled_full = pd.concat(
            [
                train_years.reset_index(drop=True),
                pd.Series(["train"] * len(train_years), name="dataset_split"),
                y_train.reset_index(drop=True),
                X_train_scaled.reset_index(drop=True),
            ],
            axis=1,
        )
        test_scaled_full = pd.concat(
            [
                test_years.reset_index(drop=True),
                pd.Series(["test"] * len(test_years), name="dataset_split"),
                y_test.reset_index(drop=True),
                X_test_scaled.reset_index(drop=True),
            ],
            axis=1,
        )
        selected_scaled_df = pd.concat(
            [train_scaled_full, test_scaled_full],
            axis=0,
            ignore_index=True,
        )
        selected_scaled_df.to_csv(
            os.path.join(feature_result_dir, "barley_features_selected_scaled_v8.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        metadata = {
            "prediction_setting": "next-year yield prediction using only previous-year or earlier information",
            "train_ratio": self.train_ratio,
            "correlation_threshold": self.correlation_threshold,
            "candidate_features": self.candidate_features,
            "selected_features": selected_features,
            "removed_features": removed_features,
        }
        with open(
            os.path.join(feature_result_dir, "feature_engineering_metadata_v8.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved v8 outputs to: {feature_result_dir}")
        return feature_result_dir

    def run(self, file_path, output_dir):
        """Run the full feature engineering workflow."""
        df = self.load_cleaned_data(file_path)
        engineered_df = self.construct_features(df)
        engineered_df, train_df, test_df = self.split_by_time(engineered_df)
        selected_features, removed_features, corr_matrix = self.select_features(train_df)
        (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            train_years,
            test_years,
        ) = self.scale_train_test(train_df, test_df, selected_features)

        return self.save_outputs(
            engineered_df=engineered_df,
            selected_features=selected_features,
            removed_features=removed_features,
            corr_matrix=corr_matrix,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            train_years=train_years,
            test_years=test_years,
            output_dir=output_dir,
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    classical_dir = os.path.dirname(script_dir)
    project_dir = os.path.dirname(classical_dir)

    parser = argparse.ArgumentParser(
        description="Build leakage-safe features for next-year barley yield prediction."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(
            project_dir,
            "artifacts",
            "classical",
            "cleaned",
            "barley_data_cleaned.csv",
        ),
        help="Path to the cleaned barley dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_dir, "artifacts", "classical", "features"),
        help="Directory for generated feature artifacts.",
    )
    args = parser.parse_args()

    pipeline = BarleyNextYearFeatureEngineering(
        train_ratio=0.8,
        correlation_threshold=0.90,
    )
    pipeline.run(file_path=args.input, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

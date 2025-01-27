from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from weather_data_loading import WeatherData


class WeatherDatasetCreator:
    """Class responsible for creating and managing weather datasets"""

    def __init__(self, data_root: Path = Path("./data")):
        self.data_root = data_root

        # Create paths for saved datasets
        self.data_root.mkdir(exist_ok=True)
        self._X_path = self.data_root / "X.csv"
        self._y_path = self.data_root / "y.csv"
        self._metadata_path = self.data_root / "metadata.csv"

    def _encode_day_of_year(self, date) -> Tuple[float, float]:
        """Create cyclic encoding of day of year"""
        day_of_year = pd.to_datetime(date).dayofyear
        year_days = 365.25  # Account for leap years

        sin_day = np.sin(2 * np.pi * day_of_year / year_days)
        cos_day = np.cos(2 * np.pi * day_of_year / year_days)

        return sin_day, cos_day

    def _get_feature_names(self, day_data: pd.DataFrame, window: int) -> List[str]:
        """Generate feature names for the window"""
        feature_names = []

        # Add cyclic encoding names for target day
        feature_names.extend(["sin_day_target", "cos_day_target"])

        # Add feature names for each day in window with day index
        base_columns = [col for col in day_data.columns if col not in ["date", "city"]]

        for i in range(window):
            for col in base_columns:
                feature_names.append(f"{col}_d{i}")

        return feature_names

    def create_complete_windowed_dataset(
        self, processed_data: pd.DataFrame, window_size: int, skip: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create complete windowed dataset with cyclic day encoding"""
        print("Creating windowed dataset...")
        print(f"Input data shape: {processed_data.shape}")
        print(f"Columns: {processed_data.columns.tolist()}")

        all_features = []
        metadata_rows = []
        y_rows = []

        # Ensure date is in datetime format
        processed_data["date"] = pd.to_datetime(processed_data["date"])

        # Process each city separately
        for city, city_df in processed_data.groupby("city"):
            print(f"\nProcessing city: {city}")
            print(f"City data shape: {city_df.shape}")

            # Sort by date
            city_df = city_df.sort_values("date").reset_index(drop=True)

            # Create windows
            for i in range(window_size, len(city_df) - skip):
                window = city_df.iloc[i - window_size : i]
                target = city_df.iloc[i + skip]

                # Validation checks
                expected_date = window["date"].iloc[-1] + pd.Timedelta(days=(1 + skip))
                if target["date"] != expected_date:
                    print(f"Date mismatch: expected {expected_date}, got {target['date']}")
                    continue

                if window.isna().any().any() or target.isna().any():
                    print(f"Found NaN values at index {i}")
                    continue

                # Feature creation
                sin_day, cos_day = self._encode_day_of_year(target["date"])
                features = [sin_day, cos_day]

                # Add window features
                window_features = window.drop(["date", "city"], axis=1).values.flatten()
                features.extend(window_features)

                # Check for any NaN or infinite values
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    print(f"Invalid values in features at index {i}")
                    continue

                all_features.append(features)
                metadata_rows.append(
                    {
                        "city": city,
                        "start_date": window["date"].iloc[0],
                        "end_date": window["date"].iloc[-1],
                        "target_date": target["date"],
                    }
                )

                # Extract target variables
                temp_mean_col = next((col for col in target.index if "temperature_mean" in col), None)
                wind_speed_col = next((col for col in target.index if "wind_speed_max" in col), None)

                if temp_mean_col is None or wind_speed_col is None:
                    print("Warning: Missing required target columns")
                    print(f"Available columns: {target.index.tolist()}")
                    continue

                y_rows.append(
                    {
                        "temperature": target[temp_mean_col],
                        "strong_wind": target[wind_speed_col] > 6,  # Threshold for strong wind
                    }
                )

        print(f"\nCreated {len(all_features)} samples")

        if not all_features:
            raise ValueError("No valid samples were created. Check the data preprocessing steps.")

        # Create DataFrames
        feature_names = self._get_feature_names(processed_data.drop(["date", "city"], axis=1), window_size)
        X_df = pd.DataFrame(np.vstack(all_features), columns=feature_names)
        y_df = pd.DataFrame(y_rows)
        metadata_df = pd.DataFrame(metadata_rows)

        print(f"Final shapes:")
        print(f"X: {X_df.shape}")
        print(f"y: {y_df.shape}")
        print(f"metadata: {metadata_df.shape}")

        return X_df, y_df, metadata_df

    def save_windowed_data(self, X_df: pd.DataFrame, y_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        """Save windowed data and feature information to CSV files"""
        # Save the main data
        X_df.to_csv(self._X_path, index=False)
        y_df.to_csv(self._y_path, index=False)

        # Save column metadata separately
        feature_metadata = pd.DataFrame({"feature_name": X_df.columns, "feature_index": range(len(X_df.columns))})
        target_metadata = pd.DataFrame({"target_name": y_df.columns, "target_index": range(len(y_df.columns))})

        # Save metadata files
        feature_metadata.to_csv(self.data_root / "feature_metadata.csv", index=False)
        target_metadata.to_csv(self.data_root / "target_metadata.csv", index=False)
        metadata_df.to_csv(self._metadata_path, index=False)

        print(f"Saved windowed data to:\n{self._X_path}\n{self._y_path}\n{self._metadata_path}")
        print(
            f"Saved metadata to:\n{self.data_root / 'feature_metadata.csv'}\n{self.data_root / 'target_metadata.csv'}"
        )

    def load_windowed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load windowed data and feature information from CSV files"""
        required_files = [
            self._X_path,
            self._y_path,
            self._metadata_path,
            self.data_root / "feature_metadata.csv",
            self.data_root / "target_metadata.csv",
        ]

        if not all(path.exists() for path in required_files):
            raise FileNotFoundError("One or more data files not found. Run pipeline with load_processed=False first.")

        # Load the raw data
        X_df = pd.read_csv(self._X_path)
        y_df = pd.read_csv(self._y_path)
        metadata_df = pd.read_csv(self._metadata_path)

        # Load and apply feature/target metadata
        feature_metadata = pd.read_csv(self.data_root / "feature_metadata.csv")
        target_metadata = pd.read_csv(self.data_root / "target_metadata.csv")

        # Order the columns according to the metadata
        feature_metadata = feature_metadata.sort_values("feature_index")
        target_metadata = target_metadata.sort_values("target_index")

        X_df = X_df[feature_metadata["feature_name"]]
        y_df = y_df[target_metadata["target_name"]]

        return X_df, y_df, metadata_df

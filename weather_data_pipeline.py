from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from weather_data_loading import WeatherDataLoader
from weather_data_merging import DataMerger
from weather_feature_pipeline import FeaturePipeline, FeatureConfig
from weather_data_collation import WeatherDatasetCreator


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation"""

    test_size: float = 0.15
    val_size: float = 0.15
    random_state: Optional[int] = 42
    window_size: int = 3
    prediction_offset: int = 1  # Days to predict ahead


class WeatherPipeline:
    """Main pipeline class for weather prediction data processing"""

    def __init__(
        self,
        data_root: Path = Path("./data"),
        feature_configs: Optional[Dict[str, FeatureConfig]] = None,
        verbose: bool = False,
    ):
        """Initialize weather pipeline.

        Args:
            data_root: Root directory for data files
            feature_configs: Optional custom feature configurations
        """
        self.data_root = data_root
        self.processed_dir = data_root / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_loader = WeatherDataLoader(data_root)
        self.data_collator = WeatherDatasetCreator(data_root)
        self.data_merger = DataMerger(verbose=verbose)
        self.feature_pipeline = FeaturePipeline(
            scaler_dir=self.processed_dir / "scalers", feature_configs=feature_configs
        )

        self.verbose = verbose

    def _encode_day_of_year(self, date) -> Tuple[float, float]:
        """Create cyclic encoding of day of year.

        Args:
            date: Date to encode

        Returns:
            Tuple of (sin_day, cos_day) values
        """
        day_of_year = pd.to_datetime(date).dayofyear
        year_days = 365.25  # Account for leap years

        sin_day = np.sin(2 * np.pi * day_of_year / year_days)
        cos_day = np.cos(2 * np.pi * day_of_year / year_days)

        return sin_day, cos_day

    def _create_windowed_samples(
        self, city_data: pd.DataFrame, window_size: int, prediction_offset: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create windowed samples for time series prediction with cyclic target day encoding.

        Args:
            city_data: DataFrame with daily weather data for a city
            window_size: Number of days to use as input
            prediction_offset: Number of days ahead to predict

        Returns:
            Tuple of (X, y_temp, y_wind) arrays
        """
        X_samples = []
        y_temp_samples = []
        y_wind_samples = []

        # Ensure date is datetime
        city_data = city_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(city_data.index):
            city_data.index = pd.to_datetime(city_data.index)

        # Create windows
        for i in range(len(city_data) - window_size - prediction_offset + 1):
            # Input window
            window = city_data.iloc[i : i + window_size]

            # Target (prediction_offset days ahead)
            target = city_data.iloc[i + window_size + prediction_offset - 1]

            # Skip if any missing data
            if window.isna().any().any() or target.isna().any():
                continue

            # Get cyclic encoding of target day
            sin_day, cos_day = self._encode_day_of_year(target.name)

            # Extract features (exclude 'city' column)
            features = window.drop("city", axis=1).values.flatten()

            # Add cyclic target day encoding to features
            features = np.concatenate([[sin_day, cos_day], features])
            X_samples.append(features)

            # Extract targets
            temp_col = next((col for col in target.index if "temperature_mean" in col), None)
            wind_col = next((col for col in target.index if "wind_speed_max" in col), None)

            if temp_col is None or wind_col is None:
                print(f"Warning: Missing required target columns at index {i}")
                print(f"Available columns: {target.index.tolist()}")
                continue

            y_temp_samples.append(target[temp_col])
            y_wind_samples.append(int(target[wind_col] > 6.0))  # Threshold for strong wind

        if not X_samples:
            raise ValueError("No valid samples could be created")

        return (np.array(X_samples), np.array(y_temp_samples), np.array(y_wind_samples))

    def _create_splits(
        self, X: np.ndarray, y_temp: np.ndarray, y_wind: np.ndarray, config: DatasetConfig
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create train/val/test splits from windowed data.

        Args:
            X: Feature matrix
            y_temp: Temperature targets
            y_wind: Wind targets
            config: Dataset configuration

        Returns:
            Dictionary containing train/val/test splits
        """
        # First split into train and temp_test
        X_train, X_temp, y_temp_train, y_temp_temp, y_wind_train, y_wind_temp = train_test_split(
            X, y_temp, y_wind, test_size=config.test_size + config.val_size, random_state=config.random_state
        )

        # Split temp_test into val and test
        val_ratio = config.val_size / (config.test_size + config.val_size)
        X_val, X_test, y_temp_val, y_temp_test, y_wind_val, y_wind_test = train_test_split(
            X_temp, y_temp_temp, y_wind_temp, test_size=0.5, random_state=config.random_state
        )

        return {
            "train": {"X": X_train, "y_temp": y_temp_train, "y_wind": y_wind_train},
            "val": {"X": X_val, "y_temp": y_temp_val, "y_wind": y_wind_val},
            "test": {"X": X_test, "y_temp": y_temp_test, "y_wind": y_wind_test},
        }

    def prepare_datasets(
        self, config: Optional[DatasetConfig] = None, load_windowed: bool = False
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """Prepare complete datasets for training."""
        config = config or DatasetConfig()

        # Define paths for windowed data
        windowed_dir = self.processed_dir / "windowed_data"
        windowed_dir.mkdir(parents=True, exist_ok=True)

        def get_windowed_paths(city: str) -> Tuple[Path, ...]:
            """Get paths for windowed data files"""
            base = windowed_dir / city
            base.mkdir(parents=True, exist_ok=True)
            return (
                base / "X.csv",  # Changed from .npy to .csv
                base / "y_temp.csv",
                base / "y_wind.csv",
                base / "feature_pipeline.pkl",  # Keep pipeline as pickle
            )

        if load_windowed:
            result = {}
            if not windowed_dir.exists():
                print("No windowed data directory found")
                load_windowed = False
            else:
                for city_dir in windowed_dir.iterdir():
                    if not city_dir.is_dir():
                        continue

                    city = city_dir.name
                    X_path, y_temp_path, y_wind_path, pipeline_path = get_windowed_paths(city)

                    if all(p.exists() for p in (X_path, y_temp_path, y_wind_path, pipeline_path)):
                        print(f"Loading pre-windowed data for {city}...")

                        # Load CSV files with appropriate column names
                        X = pd.read_csv(X_path).values
                        y_temp = pd.read_csv(y_temp_path).values.ravel()
                        y_wind = pd.read_csv(y_wind_path).values.ravel()

                        # Load the feature pipeline state
                        with open(pipeline_path, "rb") as f:
                            self.feature_pipeline = pickle.load(f)

                        # Create train/val/test splits
                        result[city] = self._create_splits(X, y_temp, y_wind, config)
                    else:
                        print(f"Warning: Incomplete data found for {city}, skipping")

            if result:
                return result
        if self.verbose:
            print("Processing raw data...")
        # Load and process raw data
        weather_data = self.data_loader.load_data()
        merged_data = self.data_merger.merge_data(weather_data)
        daily_data = self.feature_pipeline.process_data(merged_data, fit=True)

        # Create windowed datasets and splits for each city
        result = {}
        for city in daily_data["city"].unique():
            if self.verbose:
                print(f"\nProcessing {city}...")
            city_data = daily_data[daily_data["city"] == city].sort_index()

            # Create windowed samples
            X, y_temp, y_wind = self._create_windowed_samples(city_data, config.window_size, config.prediction_offset)

            # Generate feature names for X
            feature_names = self._generate_feature_names(daily_data, config.window_size)

            # Save data as CSV with proper column names
            X_path, y_temp_path, y_wind_path, pipeline_path = get_windowed_paths(city)

            # Save X with feature names
            pd.DataFrame(X, columns=feature_names).to_csv(X_path, index=False)

            # Save y values with descriptive names
            pd.DataFrame(y_temp, columns=["temperature"]).to_csv(y_temp_path, index=False)
            pd.DataFrame(y_wind, columns=["strong_wind"]).to_csv(y_wind_path, index=False)

            # Save feature pipeline state
            with open(pipeline_path, "wb") as f:
                pickle.dump(self.feature_pipeline, f)

            if self.verbose:
                print(f"Saved windowed data and feature pipeline for {city}")

            # Create splits
            result[city] = self._create_splits(X, y_temp, y_wind, config)

        return result

    def _generate_feature_names(self, daily_data: pd.DataFrame, window_size: int) -> List[str]:
        """Generate feature names for the windowed data including cyclic target day."""
        # Add cyclic target day encoding names
        feature_names = ["sin_target_day", "cos_target_day"]

        # Add names for each window position
        base_features = [col for col in daily_data.columns if col != "city"]
        for i in range(window_size):
            for feature in base_features:
                feature_names.append(f"{feature}_d{i}")

        return feature_names

    def transform_data(
        self, data: Union[pd.DataFrame, np.ndarray], inverse: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data using the feature pipeline.

        Args:
            data: Data to transform
            inverse: Whether to perform inverse transform

        Returns:
            Transformed data in the same format as input
        """
        return self.feature_pipeline.inverse_transform(data) if inverse else self.feature_pipeline.transform(data)


def main():
    """Example usage of the weather pipeline"""
    # Initialize pipeline
    feature_configs = {
        "temperature": FeatureConfig(name="temperature", agg_functions=("mean",)),
        "wind_speed": FeatureConfig(name="wind_speed", agg_functions=("max",)),
    }
    pipeline = WeatherPipeline(data_root=Path("./data"), feature_configs=feature_configs)

    # Process data and create datasets
    results = pipeline.prepare_datasets(
        config=DatasetConfig(test_size=0.15, val_size=0.15, window_size=3, prediction_offset=1, random_state=42),
        load_windowed=False,  # Set to True to load previously processed data
    )

    # Print dataset information
    for city in results.keys():
        print(f"\nCity: {city}")
        print(f"Training samples: {results[city]['train']['X'].shape[0]}")
        print(f"Validation samples: {results[city]['val']['X'].shape[0]}")
        print(f"Test samples: {results[city]['test']['X'].shape[0]}")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from pathlib import Path
import re
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass


class WeatherScalers:
    """Class to manage all scalers used in the weather prediction pipeline

    This class handles the fitting, saving, loading, and management of StandardScalers
    used for normalizing weather measurements. It maintains a cache of loaded scalers
    to avoid repeated disk reads.

    Attributes:
        scaler_dir (Path): Directory for storing scaler pickle files
        scalers (Dict[str, StandardScaler]): Cache of loaded scalers
    """

    def __init__(self, scaler_dir: Path):
        """Initialize the WeatherScalers manager

        Args:
            scaler_dir (Path): Directory where scaler files will be stored
        """
        self.scaler_dir = scaler_dir
        self.scalers: Dict[str, StandardScaler] = {}
        os.makedirs(self.scaler_dir, exist_ok=True)

    def fit_save_scaler(
        self, data: Union[np.ndarray, pd.DataFrame], name: str, columns: Optional[List[str]] = None
    ) -> StandardScaler:
        """Fit a new scaler to the data and save it to disk

        Args:
            data: Data to fit the scaler on. Can be numpy array or pandas DataFrame
            name: Identifier for the scaler
            columns: If data is a DataFrame, specify columns to fit on

        Returns:
            The fitted StandardScaler

        Raises:
            ValueError: If data format is invalid or columns don't exist
        """
        if isinstance(data, pd.DataFrame):
            if columns is None:
                raise ValueError("Must specify columns when passing a DataFrame")
            fit_data = data[columns].values
        else:
            fit_data = data if len(data.shape) > 1 else data.reshape(-1, 1)

        scaler = StandardScaler()
        scaler.fit(fit_data)

        # Save to disk
        scaler_path = self.scaler_dir / f"{name}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Cache the scaler
        self.scalers[name] = scaler
        return scaler

    def load_scaler(self, name: str) -> StandardScaler:
        """Load a scaler from disk or return cached version

        Args:
            name: Identifier of the scaler to load

        Returns:
            The loaded StandardScaler

        Raises:
            FileNotFoundError: If scaler file doesn't exist
        """
        # Return cached version if available
        if name in self.scalers:
            return self.scalers[name]

        # Load from disk
        scaler_path = self.scaler_dir / f"{name}_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"No scaler found for {name}")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            self.scalers[name] = scaler
            return scaler

    def transform_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        name: str,
        columns: Optional[List[str]] = None,
        inverse: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using a specified scaler

        Args:
            data: Data to transform
            name: Identifier of the scaler to use
            columns: If data is DataFrame, specify columns to transform
            inverse: Whether to inverse transform

        Returns:
            Transformed data in same format as input

        Raises:
            ValueError: If data format is invalid or columns don't exist
        """
        scaler = self.load_scaler(name)
        transform_func = scaler.inverse_transform if inverse else scaler.transform

        if isinstance(data, pd.DataFrame):
            if columns is None:
                raise ValueError("Must specify columns when passing a DataFrame")
            result = data.copy()
            values_to_transform = data[columns].values
            transformed_values = transform_func(values_to_transform)
            result[columns] = transformed_values
            return result
        else:
            transform_data = data if len(data.shape) > 1 else data.reshape(-1, 1)
            return transform_func(transform_data)

    def get_scaler_path(self, name: str) -> Path:
        """Get the file path for a scaler

        Args:
            name: Identifier of the scaler

        Returns:
            Path to the scaler file
        """
        return self.scaler_dir / f"{name}_scaler.pkl"

    def list_available_scalers(self) -> List[str]:
        """List all available scaler names

        Returns:
            List of scaler names that exist on disk
        """
        scaler_files = list(self.scaler_dir.glob("*_scaler.pkl"))
        return [f.stem.replace("_scaler", "") for f in scaler_files]

    def delete_scaler(self, name: str) -> None:
        """Delete a scaler from disk and cache

        Args:
            name: Identifier of the scaler to delete

        Raises:
            FileNotFoundError: If scaler doesn't exist
        """
        scaler_path = self.get_scaler_path(name)
        if not scaler_path.exists():
            raise FileNotFoundError(f"No scaler found for {name}")

        os.remove(scaler_path)
        self.scalers.pop(name, None)  # Remove from cache if present


@dataclass
class WeatherData:
    """Container for raw weather data"""

    temperature: pd.DataFrame
    pressure: pd.DataFrame
    humidity: pd.DataFrame
    wind_speed: pd.DataFrame
    wind_direction: pd.DataFrame
    weather_description: pd.DataFrame


class WeatherDataLoader:
    """Class responsible for loading raw weather data"""

    def __init__(self, data_root: Path):
        self.data_root = data_root

    def load_data(self) -> WeatherData:
        """Load all weather data files and perform initial NaN handling"""
        file_mapping = {
            "temperature": "temperature.csv",
            "pressure": "pressure.csv",
            "humidity": "humidity.csv",
            "wind_speed": "wind_speed.csv",
            "wind_direction": "wind_direction.csv",
            "weather_description": "weather_description.csv",
        }

        data_dict = {}
        for key, filename in file_mapping.items():
            file_path = self.data_root / filename
            if not file_path.exists():
                raise FileNotFoundError(f"File {filename} not found at {file_path}")

            # Load data with NaN handling
            df = pd.read_csv(file_path)

            # Convert empty strings and specified values to NaN
            df = df.replace(["", "NA", "N/A", "null", "NULL", "NaN"], np.nan)

            # For numeric columns, also convert infinite values to NaN
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

            data_dict[key] = df

        return WeatherData(**data_dict)


class WeatherPreprocessor:
    """Class responsible for preprocessing weather data"""

    def __init__(self, scaler_dir: Path):
        self.weather_scalers = WeatherScalers(scaler_dir)
        self.measurement_configs = {
            "temperature": ("temperature", ("min", "max", "mean")),
            "pressure": ("pressure", ("mean",)),
            "humidity": ("humidity", ("mean",)),
            "wind_speed": ("wind_speed", ("mean", "max")),
        }

    def _handle_nans(self, df: pd.DataFrame, measurement_name: str) -> pd.DataFrame:
        """Handle NaN values in weather data

        Args:
            df: DataFrame containing weather measurements
            measurement_name: Name of the measurement (e.g., 'temperature', 'pressure')

        Returns:
            DataFrame with handled NaN values
        """
        df = df.copy()

        # Convert datetime column
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Handle NaNs for each city separately
        city_columns = [col for col in df.columns if col != "datetime"]

        for city in city_columns:
            # Sort by datetime and set it as index for time-based operations
            df = df.sort_values("datetime")
            temp_df = df.set_index("datetime")

            # Forward fill followed by backward fill for short gaps (up to 3 hours)
            temp_df[city] = temp_df[city].ffill(limit=3)
            temp_df[city] = temp_df[city].bfill(limit=3)

            # For remaining NaNs, use rolling window interpolation
            window_size = 24  # 24-hour window
            temp_df[city] = temp_df[city].interpolate(method="time", limit=window_size, limit_direction="both")

            # Reset index to get datetime back as a column
            df = temp_df.reset_index()

            # For any remaining NaNs at the start/end, use seasonal patterns
            if df[city].isna().any():
                # Calculate hourly and daily patterns
                df["hour"] = df["datetime"].dt.hour
                df["dayofweek"] = df["datetime"].dt.dayofweek

                # Group by hour and day of week to get typical values
                patterns = df.groupby(["hour", "dayofweek"])[city].mean()

                # Fill remaining NaNs with typical values for that hour and day
                for idx in df[df[city].isna()].index:
                    hour = df.loc[idx, "hour"]
                    day = df.loc[idx, "dayofweek"]
                    if (hour, day) in patterns:
                        df.loc[idx, city] = patterns[hour, day]

                # Drop temporary columns
                df = df.drop(["hour", "dayofweek"], axis=1)

        # Log remaining NaNs if any
        remaining_nans = df[city_columns].isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain in {measurement_name} data")

        return df

    def _handle_categorical_nans(self, df: pd.DataFrame, measurement_name: str) -> pd.DataFrame:
        """Handle NaN values in categorical weather data

        Args:
            df: DataFrame containing categorical measurements (e.g., weather descriptions)
            measurement_name: Name of the measurement

        Returns:
            DataFrame with handled NaN values
        """
        # Create a deep copy to avoid modifying the original
        result_df = df.copy()

        # Convert datetime column
        result_df["datetime"] = pd.to_datetime(result_df["datetime"])

        # Handle NaNs for each city separately
        city_columns = [col for col in result_df.columns if col != "datetime"]

        # Sort by datetime once
        result_df = result_df.sort_values("datetime")

        for city in city_columns:
            # Create a Series with datetime index for this city
            city_series = pd.Series(result_df[city].values, index=result_df["datetime"], name=city)

            # Forward fill followed by backward fill for short gaps (up to 3 hours)
            city_series = city_series.ffill(limit=3)
            city_series = city_series.bfill(limit=3)

            # For remaining NaNs, use mode (most common value) based on hour and day
            if city_series.isna().any():
                # Create hour and day series aligned with our data
                hours = city_series.index.hour
                days = city_series.index.dayofweek

                # Get indices of remaining NaNs
                nan_indices = city_series.index[city_series.isna()]

                for idx in nan_indices:
                    hour = idx.hour
                    day = idx.dayofweek

                    # Get most common value for this hour and day
                    mask = (hours == hour) & (days == day) & (~city_series.isna())
                    matching_values = city_series[mask]

                    if not matching_values.empty:
                        # Get mode of matching values
                        mode_value = matching_values.mode()
                        if not mode_value.empty:
                            city_series.loc[idx] = mode_value.iloc[0]
                    else:
                        # If no mode found for specific hour/day, use overall mode
                        overall_mode = city_series.dropna().mode()
                        if not overall_mode.empty:
                            city_series.loc[idx] = overall_mode.iloc[0]

            # Update the original dataframe using loc to ensure proper alignment
            result_df.loc[:, city] = city_series.values

        # Log remaining NaNs if any
        remaining_nans = result_df[city_columns].isna().sum().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain in {measurement_name} data")

        return result_df

    def preprocess_data(self, weather_data: WeatherData, is_training: bool = True) -> pd.DataFrame:
        """Preprocess all weather data components with NaN handling"""
        transformed_dfs = []

        # Handle NaNs and transform scalar measurements
        for data_key, (measurement_name, agg_types) in self.measurement_configs.items():
            df = getattr(weather_data, data_key)
            # Handle NaNs before any transformations
            df = self._handle_nans(df, measurement_name)
            transformed_df = self._transform_scalar_data(
                df, measurement_name, agg_types, self.weather_scalers, is_training
            )
            transformed_dfs.append(transformed_df)

        # Handle NaNs in wind direction before transformation
        wind_dir_df = self._handle_nans(weather_data.wind_direction, "wind_direction")
        transformed_dfs.append(self._transform_wind_direction(wind_dir_df))

        # Handle NaNs in weather description using categorical method
        weather_desc_df = self._handle_categorical_nans(weather_data.weather_description, "weather_description")
        transformed_dfs.append(self._transform_weather_description(weather_desc_df))

        # Merge all dataframes
        result = transformed_dfs[0]
        for df in transformed_dfs[1:]:
            result = pd.merge(result, df, on=["date", "city"], how="outer")

        # Final check for any remaining NaNs
        remaining_nans = result.isna().sum()
        if remaining_nans.any():
            print("Warning: Remaining NaN values in processed data:")
            print(remaining_nans[remaining_nans > 0])

        return result.sort_values(["date", "city"])

    @staticmethod
    def _transform_scalar_data(
        df: pd.DataFrame,
        measurement_name: str,
        agg_types: tuple,
        weather_scalers: "WeatherScalers",
        is_training: bool = True,
    ) -> pd.DataFrame:
        """Transform scalar weather data"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        city_columns = [col for col in df.columns if col != "datetime"]
        df_melted = df.melt(id_vars=["datetime"], value_vars=city_columns, var_name="city", value_name=measurement_name)

        df_melted["date"] = df_melted["datetime"].dt.date

        result = (
            df_melted.groupby(["date", "city"])
            .agg({measurement_name: [(f"{measurement_name}_{agg_type}", agg_type) for agg_type in agg_types]})
            .reset_index()
        )

        result.columns = ["date", "city"] + [f"{measurement_name}_{agg_type}" for agg_type in agg_types]

        cols_to_normalize = [f"{measurement_name}_{agg_type}" for agg_type in agg_types]

        if is_training:
            scaler = weather_scalers.fit_save_scaler(result[cols_to_normalize].values, measurement_name)
        else:
            scaler = weather_scalers.load_scaler(measurement_name)

        values_to_transform = result[cols_to_normalize].values
        transformed_values = scaler.transform(values_to_transform)
        result[cols_to_normalize] = transformed_values
        return result

    @staticmethod
    def _transform_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
        """Transform wind direction data"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        city_columns = [col for col in df.columns if col != "datetime"]
        df_melted = df.melt(id_vars=["datetime"], value_vars=city_columns, var_name="city", value_name="wind_direction")

        radians = np.deg2rad(df_melted["wind_direction"])
        df_melted["wind_direction_x"] = np.cos(radians)
        df_melted["wind_direction_y"] = np.sin(radians)

        df_melted["date"] = df_melted["datetime"].dt.date

        return (
            df_melted.groupby(["date", "city"])
            .agg({"wind_direction_x": "mean", "wind_direction_y": "mean"})
            .reset_index()
        )

    @staticmethod
    def _to_snake_case(text: str) -> str:
        # List of words to remove (can be expanded)
        connecting_words = {"with", "and", "or", "the", "a", "an", "in", "at", "on"}

        # Replace special characters with spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))

        # Convert to lowercase and split
        words = text.lower().split()

        # Filter out connecting words
        words = [word for word in words if word not in connecting_words]

        # Join with underscore
        return "_".join(words)

    @staticmethod
    def _get_weather_category_mapping():
        # Sky condition mappings
        sky_clear = ["sky_is_clear"]
        sky_partial = ["few_clouds", "scattered_clouds", "broken_clouds"]
        sky_covered = ["overcast_clouds", "mist", "fog", "haze", "smoke", "dust", "sand", "volcanic_ash"]

        # Precipitation mappings
        precip_light = [
            "light_rain",
            "light_snow",
            "light_intensity_drizzle",
            "drizzle",
            "light_intensity_drizzle_rain",
            "light_intensity_shower_rain",
            "light_rain_snow",
            "light_shower_sleet",
            "light_shower_snow",
        ]
        precip_heavy = [
            "heavy_intensity_rain",
            "very_heavy_rain",
            "heavy_intensity_drizzle",
            "heavy_intensity_shower_rain",
            "heavy_shower_snow",
            "heavy_snow",
            "shower_rain",
            "shower_drizzle",
            "shower_snow",
            "rain_snow",
            "moderate_rain",
            "freezing_rain",
        ]

        # Storm mappings
        storm = [
            "thunderstorm",
            "heavy_thunderstorm",
            "ragged_thunderstorm",
            "thunderstorm_drizzle",
            "thunderstorm_heavy_drizzle",
            "thunderstorm_heavy_rain",
            "thunderstorm_light_drizzle",
            "thunderstorm_light_rain",
            "thunderstorm_rain",
            "proximity_thunderstorm",
            "proximity_thunderstorm_drizzle",
            "proximity_thunderstorm_rain",
            "squalls",
            "tornado",
        ]

        # Create the mapping dictionary
        category_mapping = {}

        # Sky condition
        for condition in sky_clear:
            category_mapping[condition] = "sky_clear"
        for condition in sky_partial:
            category_mapping[condition] = "sky_partial"
        for condition in sky_covered:
            category_mapping[condition] = "sky_covered"

        # Precipitation
        for condition in precip_light:
            category_mapping[condition] = "precip_light"
        for condition in precip_heavy:
            category_mapping[condition] = "precip_heavy"

        # Storm
        for condition in storm:
            category_mapping[condition] = "storm"

        return category_mapping

    @staticmethod
    def _transform_weather_description(df: pd.DataFrame) -> pd.DataFrame:
        """Transform weather description data"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        city_columns = [col for col in df.columns if col != "datetime"]
        df_melted = df.melt(id_vars=["datetime"], value_vars=city_columns, var_name="city", value_name="class_label")

        df_melted["date"] = df_melted["datetime"].dt.date
        df_melted["class_label"] = df_melted["class_label"].apply(WeatherPreprocessor._to_snake_case)

        category_mapping = WeatherPreprocessor._get_weather_category_mapping()
        df_melted["weather_category"] = df_melted["class_label"].map(category_mapping)

        one_hot = pd.get_dummies(df_melted["weather_category"], prefix="weather_description")
        df_melted = pd.concat([df_melted[["date", "city"]], one_hot], axis=1)

        return df_melted.groupby(["date", "city"]).mean().reset_index()


class WeatherDatasetCreator:
    """Class responsible for creating and managing weather datasets"""

    def __init__(self, weather_scalers: WeatherScalers, data_root: Path = Path("./data")):
        self.weather_scalers = weather_scalers
        self.data_root = data_root

        # Create paths for saved datasets
        self.data_root.mkdir(exist_ok=True)
        self._X_path = self.data_root / "X.csv"
        self._y_path = self.data_root / "y.csv"
        self._metadata_path = self.data_root / "metadata.csv"

    def _encode_day_of_year(self, date):
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
        for i in range(len(window)):
            for col in day_data.columns:
                feature_names.append(f"{col}_d{i}")
        return feature_names

    def create_complete_windowed_dataset(
        self, processed_data: pd.DataFrame, window_size: int, skip: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create complete windowed dataset with cyclic day encoding"""
        all_features = []
        metadata_rows = []
        y_rows = []
        scaled_speed_threshold = self.weather_scalers.transform_data(np.array([[6, 6]]), "wind_speed")[0][1]
        for city, city_df in processed_data.groupby("city"):
            city_df = city_df.reset_index(drop=True)

            for i in range(window_size, len(city_df) - skip):
                window = city_df.iloc[i - window_size : i]
                target = city_df.iloc[i + skip]

                # Validation checks
                expected_date = pd.to_datetime(window["date"].iloc[-1]) + pd.Timedelta(days=(1 + skip))
                if pd.to_datetime(target["date"]) != expected_date:
                    continue

                if window.isna().any().any() or target.isna().any():
                    continue

                # Feature creation
                sin_day, cos_day = self._encode_day_of_year(target["date"])
                features = [sin_day, cos_day]
                features.extend(window.drop(["date", "city"], axis=1).values.flatten())

                all_features.append(features)
                metadata_rows.append(
                    {
                        "city": city,
                        "start_date": window["date"].iloc[0],
                        "end_date": window["date"].iloc[-1],
                        "target_date": target["date"],
                    }
                )
                y_rows.append(
                    {
                        "temperature": target["temperature_mean"],
                        "strong_wind": target["wind_speed_max"] >= scaled_speed_threshold,
                    }
                )

        # Create DataFrames
        feature_names = self._get_feature_names(processed_data.drop(["date", "city"], axis=1), window)
        X_df = pd.DataFrame(np.vstack(all_features), columns=feature_names)
        y_df = pd.DataFrame(y_rows)
        metadata_df = pd.DataFrame(metadata_rows)

        return X_df, y_df, metadata_df

    def save_windowed_data(self, X_df: pd.DataFrame, y_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
        """Save windowed data to CSV files"""
        X_df.to_csv(self._X_path, index=False)
        y_df.to_csv(self._y_path, index=False)
        metadata_df.to_csv(self._metadata_path, index=False)
        print(f"Saved windowed data to:\n{self._X_path}\n{self._y_path}\n{self._metadata_path}")

    def load_windowed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load windowed data from CSV files"""
        if not all(path.exists() for path in [self._X_path, self._y_path, self._metadata_path]):
            raise FileNotFoundError(
                "One or more windowed data files not found. Run pipeline with load_processed=False first."
            )

        X_df = pd.read_csv(self._X_path)
        y_df = pd.read_csv(self._y_path)
        metadata_df = pd.read_csv(self._metadata_path)

        return X_df, y_df, metadata_df

    def split_datasets(
        self,
        X_df: pd.DataFrame,
        y_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        test_size: float,
        val_size: float,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """Split windowed data into train/val/test sets by city"""
        train_data, val_data, test_data = {}, {}, {}

        for city in metadata_df["city"].unique():
            city_mask = metadata_df["city"] == city
            if not any(city_mask):
                continue

            X_city = X_df.loc[city_mask].values
            y_temp_city = y_df.loc[city_mask, "temperature"].values
            y_wind_city = y_df.loc[city_mask, "strong_wind"].values

            # Split into train/val/test
            X_temp, X_test, y_temp_temp, y_temp_test, y_wind_temp, y_wind_test = train_test_split(
                X_city, y_temp_city, y_wind_city, test_size=test_size, random_state=random_state
            )

            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_temp_train, y_temp_val, y_wind_train, y_wind_val = train_test_split(
                X_temp, y_temp_temp, y_wind_temp, test_size=val_size_adjusted, random_state=random_state
            )

            # Store splits
            train_data[city] = {"X": X_train, "y_temp": y_temp_train, "y_wind": y_wind_train}
            val_data[city] = {"X": X_val, "y_temp": y_temp_val, "y_wind": y_wind_val}
            test_data[city] = {"X": X_test, "y_temp": y_temp_test, "y_wind": y_wind_test}

        return train_data, val_data, test_data


class WeatherPipeline:
    """Main pipeline class for weather prediction"""

    def __init__(self, data_root: Path = Path("./data"), scaler_dir: Path = Path("./scalers")):
        self.data_root = data_root
        self.data_loader = WeatherDataLoader(data_root)
        self.preprocessor = WeatherPreprocessor(scaler_dir)
        self.dataset_creator = WeatherDatasetCreator(self.preprocessor.weather_scalers, data_root)

    def prepare_datasets(
        self,
        window_size: int = 3,
        skip: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
        load_processed: bool = False,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """Prepare complete datasets for training"""
        if load_processed:
            # Load pre-saved windowed data
            X_df, y_df, metadata_df = self.dataset_creator.load_windowed_data()
        else:
            # Create and save windowed data
            weather_data = self.data_loader.load_data()
            processed_data = self.preprocessor.preprocess_data(weather_data, is_training=True)
            X_df, y_df, metadata_df = self.dataset_creator.create_complete_windowed_dataset(
                processed_data, window_size, skip
            )
            self.dataset_creator.save_windowed_data(X_df, y_df, metadata_df)

        # Split the data into train/val/test sets
        return self.dataset_creator.split_datasets(X_df, y_df, metadata_df, test_size, val_size, random_state)

    def prepare_prediction_data(
        self,
        window_size: int = 3,
        skip: int = 1,
        load_processed: bool = False,
    ) -> np.ndarray:
        """Prepare data for prediction"""
        if load_processed:
            X_df, _, _ = self.dataset_creator.load_windowed_data()
            return X_df.values
        else:
            weather_data = self.data_loader.load_data()
            processed_data = self.preprocessor.preprocess_data(weather_data, is_training=False)
            X_df, _, _ = self.dataset_creator.create_complete_windowed_dataset(processed_data, window_size, skip)
            return X_df.values


# Example usage:
if __name__ == "__main__":
    pipeline = WeatherPipeline()

    # First run: process data and save windowed datasets
    train_data, val_data, test_data = pipeline.prepare_datasets(
        window_size=3,
        skip=1,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        load_processed=False,  # Process raw data and save windowed datasets
    )

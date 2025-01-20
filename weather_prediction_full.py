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
        """Load all weather data files"""
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
            data_dict[key] = pd.read_csv(file_path)

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

    def preprocess_data(self, weather_data: WeatherData, is_training: bool = True) -> pd.DataFrame:
        """Preprocess all weather data components"""
        transformed_dfs = []

        # Transform scalar measurements
        for data_key, (measurement_name, agg_types) in self.measurement_configs.items():
            df = getattr(weather_data, data_key)
            transformed_df = self._transform_scalar_data(
                df, measurement_name, agg_types, self.weather_scalers, is_training
            )
            transformed_dfs.append(transformed_df)

        # Transform wind direction
        transformed_dfs.append(self._transform_wind_direction(weather_data.wind_direction))

        # Transform weather description
        transformed_dfs.append(self._transform_weather_description(weather_data.weather_description))

        # Merge all dataframes
        result = transformed_dfs[0]
        for df in transformed_dfs[1:]:
            result = pd.merge(result, df, on=["date", "city"], how="outer")

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
    """Class responsible for creating training/validation/test datasets"""

    def __init__(self, weather_scalers: WeatherScalers):
        self.weather_scalers = weather_scalers

    def create_datasets(
        self,
        df: pd.DataFrame,
        window_size: int = 3,
        skip: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create train/validation/test datasets"""
        X, y_temp, y_wind = self._create_prediction_windows(df, window_size, skip)

        # First split: separate test set
        X_temp, X_test, y_temp_temp, y_temp_test, y_wind_temp, y_wind_test = train_test_split(
            X, y_temp, y_wind, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_temp_train, y_temp_val, y_wind_train, y_wind_val = train_test_split(
            X_temp, y_temp_temp, y_wind_temp, test_size=val_size_adjusted, random_state=random_state
        )

        datasets = {
            "train": {"X": X_train, "y_temp": y_temp_train, "y_wind": y_wind_train},
            "val": {"X": X_val, "y_temp": y_temp_val, "y_wind": y_wind_val},
            "test": {"X": X_test, "y_temp": y_temp_test, "y_wind": y_wind_test},
        }

        return datasets["train"], datasets["val"], datasets["test"]

    def _create_prediction_windows(
        self, df: pd.DataFrame, window_size: int = 3, skip: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create prediction windows"""
        wind_speed_scaler = self.weather_scalers.load_scaler("wind_speed")
        temp_scaler = self.weather_scalers.load_scaler("temperature")

        wind_speed_limit = wind_speed_scaler.transform([[6, 6]])[0][1]  # mean, max
        df = df.copy()
        df["strong_wind"] = (df["wind_speed_max"] >= wind_speed_limit).astype(int)

        X, y_temp, y_wind = [], [], []

        for i in range(window_size, len(df) - skip + 1):
            window = df.iloc[i - window_size : i]
            target = df.iloc[i + skip - 1]

            expected_date = pd.to_datetime(window["date"].iloc[-1]) + pd.Timedelta(days=1)
            if pd.to_datetime(target["date"]) != expected_date:
                continue

            features = window.drop(["date", "city", "wind_speed_max"], axis=1).values.flatten()
            X.append(features)

            # Fix: Create a properly shaped array for inverse transform
            temp_values = [[target["temperature_min"], target["temperature_max"], target["temperature_mean"]]]
            original_temp = temp_scaler.inverse_transform(temp_values)[0][2]  # Get mean temperature
            y_temp.append(original_temp)
            y_wind.append(target["strong_wind"])

        return np.array(X), np.array(y_temp), np.array(y_wind)


class WeatherPipeline:
    """Main pipeline class for weather prediction"""

    def __init__(self, data_root: Path = Path("./data"), scaler_dir: Path = Path("./scalers")):
        self.data_loader = WeatherDataLoader(data_root)
        self.preprocessor = WeatherPreprocessor(scaler_dir)
        self.dataset_creator = WeatherDatasetCreator(self.preprocessor.weather_scalers)

    def prepare_datasets(
        self,
        window_size: int = 3,
        skip: int = 1,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Prepare complete datasets for training"""
        # Load raw data
        weather_data = self.data_loader.load_data()

        # Preprocess data
        processed_data = self.preprocessor.preprocess_data(weather_data, is_training=True)

        # Create datasets
        return self.dataset_creator.create_datasets(
            processed_data,
            window_size=window_size,
            skip=skip,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

    def prepare_prediction_data(self, window_size: int = 3, skip: int = 1) -> np.ndarray:
        """Prepare data for prediction"""
        weather_data = self.data_loader.load_data()
        processed_data = self.preprocessor.preprocess_data(weather_data, is_training=False)
        X, _, _ = self.dataset_creator._create_prediction_windows(processed_data, window_size, skip)
        return X


# Example usage:
if __name__ == "__main__":
    pipeline = WeatherPipeline(Path("./NN-project-3/data"))

    # Prepare datasets for training
    train_data, val_data, test_data = pipeline.prepare_datasets(
        window_size=3, skip=1, test_size=0.2, val_size=0.2, random_state=42
    )

    # Access the data
    X_train, y_temp_train, y_wind_train = train_data["X"], train_data["y_temp"], train_data["y_wind"]
    X_val, y_temp_val, y_wind_val = val_data["X"], val_data["y_temp"], val_data["y_wind"]
    X_test, y_temp_test, y_wind_test = test_data["X"], test_data["y_temp"], test_data["y_wind"]

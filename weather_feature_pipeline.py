from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""

    name: str
    agg_functions: Tuple[str, ...]
    cyclic: bool = False
    should_scale: bool = True


class FeatureTransformer(ABC):
    """Base class for feature transformations"""

    def __init__(self, config: FeatureConfig):
        self.config = config

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit transformer to data"""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Transform data"""
        pass


class CyclicTransformer(FeatureTransformer):
    """Transform cyclic data (like angles) to sin/cos components"""

    def __init__(self, config: FeatureConfig):
        super().__init__(config)

    def fit(self, data: pd.DataFrame) -> None:
        """No fitting needed for cyclic transformation"""
        pass

    def transform(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Transform angles to sin/cos components, returning separate DataFrames"""
        # Convert to radians if the data appears to be in degrees (values > 2π)
        max_val = data.abs().max().max()
        angles_rad = np.deg2rad(data) if max_val > 2 * np.pi else data

        # Calculate sin and cos components
        sin_df = pd.DataFrame(np.sin(angles_rad), index=data.index, columns=data.columns)
        cos_df = pd.DataFrame(np.cos(angles_rad), index=data.index, columns=data.columns)

        return {f"{self.config.name}_sin": sin_df, f"{self.config.name}_cos": cos_df}


class StandardScalingTransformer(FeatureTransformer):
    """Standardize numerical features"""

    def __init__(self, config: FeatureConfig, scaler_dir: Path):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.scaler_dir = scaler_dir
        self.scaler_path = self.scaler_dir / f"{config.name}_scaler.pkl"
        self.is_fit = False

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Fit scaler to all data values"""
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            values = data.values
        else:
            values = data

        # Flatten and reshape for fitting if needed
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])

        self.scaler.fit(values)
        self.is_fit = True

        # Save scaler
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def transform(
        self, data: Union[pd.DataFrame, np.ndarray], inverse: bool = False
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data using fitted scaler"""
        if not self.is_fit:
            if not self.scaler_path.exists():
                raise ValueError(f"No saved scaler found for {self.config.name}")
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
                self.is_fit = True

        # Handle different input types
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            values = data.values
        else:
            values = data

        # Reshape if needed
        original_shape = values.shape
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.ndim > 2:
            values = values.reshape(-1, values.shape[-1])

        # Transform
        if inverse:
            transformed = self.scaler.inverse_transform(values)
        else:
            transformed = self.scaler.transform(values)

        # Restore original shape if input was not 2D
        if values.ndim != 2:
            transformed = transformed.reshape(original_shape)

        # Return same type as input
        if is_dataframe:
            return pd.DataFrame(transformed, index=data.index, columns=data.columns)
        return transformed


class FeaturePipeline:
    """Pipeline for extracting daily features from hourly data"""

    def __init__(self, scaler_dir: Path, feature_configs: Optional[Dict[str, FeatureConfig]] = None):
        self.scaler_dir = scaler_dir
        self.feature_configs = feature_configs or self._default_feature_configs()
        self.transformers = self._initialize_transformers()

    def _default_feature_configs(self) -> Dict[str, FeatureConfig]:
        """Default feature configuration if none provided"""
        return {
            "temperature": FeatureConfig(name="temperature", agg_functions=("mean", "min", "max"), should_scale=True),
            "wind_speed": FeatureConfig(name="wind_speed", agg_functions=("mean", "max"), should_scale=True),
            "wind_direction": FeatureConfig(
                name="wind_direction", agg_functions=("mean",), cyclic=True, should_scale=False
            ),
            "pressure": FeatureConfig(name="pressure", agg_functions=("mean",), should_scale=True),
            "humidity": FeatureConfig(name="humidity", agg_functions=("mean", "min", "max"), should_scale=True),
        }

    def _initialize_transformers(self) -> Dict[str, FeatureTransformer]:
        """Initialize transformers based on configuration"""
        transformers = {}
        for name, config in self.feature_configs.items():
            if config.cyclic:
                transformers[name] = CyclicTransformer(config)
            elif config.should_scale:
                transformers[name] = StandardScalingTransformer(config, self.scaler_dir)
        return transformers

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """Fit transformers on hourly data"""
        for name, transformer in self.transformers.items():
            if name in data:
                transformer.fit(data[name])

    def transform_hourly(self, data: Dict[str, pd.DataFrame], skip_scaling: bool = False) -> Dict[str, pd.DataFrame]:
        """Transform hourly data"""
        transformed = {}
        # Transform each measurement
        for name, df in data.items():
            if name not in self.transformers:
                transformed[name] = df  # Keep original if no transformer
                continue

            transformer = self.transformers[name]

            # Skip scaling transformers if requested
            if skip_scaling and isinstance(transformer, StandardScalingTransformer):
                transformed[name] = df
                continue

            result = transformer.transform(df)

            # Handle cyclic transformers that return multiple DataFrames
            if isinstance(result, dict):
                transformed.update(result)
            else:
                transformed[name] = result

        return transformed

    def aggregate_daily(self, hourly_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate hourly data to daily statistics"""
        daily_stats = []

        # Get base measurements and their configs
        base_measurements = {
            name: config
            for name, config in self.feature_configs.items()
            if name in hourly_data or (config.cyclic and f"{name}_sin" in hourly_data and f"{name}_cos" in hourly_data)
        }

        # Process each city
        first_data = next(iter(hourly_data.values()))
        city_names = first_data.columns

        for city in city_names:
            city_stats = {}

            # Process each measurement
            for measurement, config in base_measurements.items():
                if config.cyclic:
                    # Handle cyclic data (wind direction)
                    sin_data = hourly_data[f"{measurement}_sin"][city]
                    cos_data = hourly_data[f"{measurement}_cos"][city]

                    # Average the components
                    sin_mean = sin_data.resample("D").mean()
                    cos_mean = cos_data.resample("D").mean()

                    city_stats[f"{measurement}_sin"] = sin_mean
                    city_stats[f"{measurement}_cos"] = cos_mean
                else:
                    # Regular aggregation for non-cyclic data
                    city_data = hourly_data[measurement][city]
                    for agg_func in config.agg_functions:
                        agg_data = getattr(city_data.resample("D"), agg_func)()
                        city_stats[f"{measurement}_{agg_func}"] = agg_data

            # Combine all stats for the city
            city_df = pd.DataFrame(city_stats)
            city_df["city"] = city
            daily_stats.append(city_df)

        # Combine all cities
        result = pd.concat(daily_stats)
        return result.sort_index()

    def process_data(self, data: Dict[str, pd.DataFrame], fit: bool = False) -> pd.DataFrame:
        """Complete feature extraction pipeline"""

        if fit:
            # Fit scalers on raw data before any transformations
            for name, transformer in self.transformers.items():
                if name in data and isinstance(transformer, StandardScalingTransformer):
                    transformer.fit(data[name])

        # Transform hourly data (skip scaling during dataset creation)
        hourly_transformed = self.transform_hourly(data, skip_scaling=True)

        # Aggregate to daily statistics
        daily_stats = self.aggregate_daily(hourly_transformed)

        return daily_stats

    def scale_features(
        self, data: Union[pd.DataFrame, np.ndarray], feature_names: List[str]
    ) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            return self._scale_dataframe(data)
        else:
            return self._scale_array(data, feature_names)

    def _scale_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features in a DataFrame using column names"""
        result = data.copy()

        for name, config in self.feature_configs.items():
            if not config.should_scale or name not in self.transformers:
                continue

            # Find columns for this feature
            feature_cols = [col for col in result.columns if col.startswith(f"{name}_") and col != "city"]

            if not feature_cols:
                continue

            # Get the transformer
            transformer = self.transformers[name]
            if isinstance(transformer, StandardScalingTransformer):
                result[feature_cols] = transformer.transform(result[feature_cols])

        return result

    def _scale_array(self, data: np.ndarray, column_names: List[str]) -> np.ndarray:
        """Scale features based on column names using pre-fit scalers"""
        result = data.copy()

        for name, config in self.feature_configs.items():
            if not config.should_scale or name not in self.transformers:
                continue

            # Find columns for this feature type
            feature_cols = [i for i, col in enumerate(column_names) if name in col]
            if feature_cols:
                transformer = self.transformers[name]
                if isinstance(transformer, StandardScalingTransformer):
                    result[:, feature_cols] = transformer.transform(data[:, feature_cols])

        return result

    def inverse_scale_features(
        self, data: Union[pd.DataFrame, np.ndarray], feature_type: Optional[str] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Inverse scale features that were scaled.

        Args:
            data: Input data as either DataFrame or numpy array
            feature_type: If data is numpy array, specify which feature type to inverse scale

        Returns:
            Inverse scaled data in same format as input
        """
        if isinstance(data, pd.DataFrame):
            return self._inverse_scale_dataframe(data)
        else:
            return self._inverse_scale_array(data, feature_type)

    def _inverse_scale_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse scale features in a DataFrame using column names"""
        result = data.copy()

        for name, config in self.feature_configs.items():
            if not config.should_scale or name not in self.transformers:
                continue

            feature_cols = [col for col in result.columns if col.startswith(f"{name}_") and col != "city"]

            if not feature_cols:
                continue

            transformer = self.transformers[name]
            if isinstance(transformer, StandardScalingTransformer):
                result[feature_cols] = transformer.transform(result[feature_cols], inverse=True)

        return result

    def _inverse_scale_array(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """Inverse scale features in a numpy array for a specific feature type"""
        if feature_type not in self.feature_configs:
            raise ValueError(f"Unknown feature type: {feature_type}")

        config = self.feature_configs[feature_type]
        if not config.should_scale or feature_type not in self.transformers:
            return data

        transformer = self.transformers[feature_type]
        if isinstance(transformer, StandardScalingTransformer):
            return transformer.transform(data, inverse=True)

        return data

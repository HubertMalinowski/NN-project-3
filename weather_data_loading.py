from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass


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

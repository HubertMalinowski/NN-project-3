from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict
import dataclasses
from weather_data_loading import WeatherData


class DataMerger:
    """Handles initial data loading, merging and cleaning of weather measurements"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _handle_numeric_missing_values(self, df: pd.DataFrame, measurement: str) -> pd.DataFrame:
        """Handle missing values in numeric weather data using interpolation and seasonal patterns"""
        df = df.copy()

        # Ensure datetime index
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)

        # Process each city
        for city in df.columns:
            # Sort by datetime
            city_data = df[city].sort_index()

            # Short-term interpolation (up to 3 hours)
            city_data = city_data.interpolate(method="linear", limit=3)

            # Medium-term interpolation (up to 24 hours)
            if city_data.isna().any():
                # Use time-based interpolation for longer gaps
                city_data = city_data.interpolate(method="time", limit=24)

            # Long-term pattern-based filling
            if city_data.isna().any():
                # Group by hour and month for seasonal patterns
                city_data = city_data.to_frame("value")
                city_data["hour"] = city_data.index.hour
                city_data["month"] = city_data.index.month

                # Calculate patterns based on month and hour
                patterns = city_data.groupby(["month", "hour"])["value"].mean()

                # Fill remaining NaNs with typical values for that hour and month
                for idx in city_data[city_data["value"].isna()].index:
                    hour = idx.hour
                    month = idx.month
                    if (month, hour) in patterns.index:
                        city_data.loc[idx, "value"] = patterns.loc[(month, hour)]

                # If still have NaNs, try using only monthly patterns
                if city_data["value"].isna().any():
                    monthly_patterns = city_data.groupby("month")["value"].mean()
                    for idx in city_data[city_data["value"].isna()].index:
                        month = idx.month
                        if month in monthly_patterns.index:
                            city_data.loc[idx, "value"] = monthly_patterns[month]

                df[city] = city_data["value"]

            else:
                df[city] = city_data

        return df

    def _handle_categorical_missing_values(self, df: pd.DataFrame, measurement: str) -> pd.DataFrame:
        """Handle missing values in categorical weather data using mode and seasonal patterns"""
        df = df.copy()

        # Ensure datetime index
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)

        # Process each city
        for city in df.columns:
            # Sort by datetime
            city_data = df[city].copy()

            # Forward fill for short gaps (up to 3 hours)
            city_data = city_data.ffill(limit=3)

            # Backward fill for remaining short gaps
            city_data = city_data.bfill(limit=3)

            # For remaining NaNs, use seasonal patterns
            if city_data.isna().any():
                # Convert to DataFrame for easier handling
                city_df = city_data.to_frame("value")
                city_df["hour"] = city_df.index.hour
                city_df["month"] = city_df.index.month

                # Calculate most common values by month and hour
                patterns = city_df.groupby(["month", "hour"])["value"].agg(
                    lambda x: x.mode().iloc[0] if not x.empty else np.nan
                )

                # Fill NaNs using hour-month patterns
                for idx in city_df[city_df["value"].isna()].index:
                    hour = idx.hour
                    month = idx.month
                    if (month, hour) in patterns.index:
                        city_df.loc[idx, "value"] = patterns.loc[(month, hour)]

                # If still have NaNs, try using only monthly patterns
                if city_df["value"].isna().any():
                    monthly_patterns = city_df.groupby("month")["value"].agg(
                        lambda x: x.mode().iloc[0] if not x.empty else np.nan
                    )
                    for idx in city_df[city_df["value"].isna()].index:
                        month = idx.month
                        if month in monthly_patterns.index:
                            city_df.loc[idx, "value"] = monthly_patterns[month]

                # If any remaining NaNs, use overall mode
                if city_df["value"].isna().any():
                    overall_mode = city_df["value"].mode().iloc[0]
                    city_df["value"] = city_df["value"].fillna(overall_mode)

                df[city] = city_df["value"]
            else:
                df[city] = city_data

        return df

    def merge_data(self, weather_data: WeatherData) -> Dict[str, pd.DataFrame]:
        """Merge and clean all weather measurements"""
        merged_data = {}

        # Define which measurements are categorical
        categorical_measurements = {"weather_description"}

        # Process all available measurements
        for attr_name in dir(weather_data):
            # Skip private attributes and non-DataFrame attributes
            if attr_name.startswith("_"):
                continue

            attr = getattr(weather_data, attr_name)
            if not isinstance(attr, pd.DataFrame):
                continue

            if self.verbose:
                print(f"Processing {attr_name} data...")

            # Choose appropriate missing value handling method
            if attr_name in categorical_measurements:
                cleaned_df = self._handle_categorical_missing_values(attr, attr_name)
            else:
                cleaned_df = self._handle_numeric_missing_values(attr, attr_name)

            merged_data[attr_name] = cleaned_df

            # Log remaining missing values if any
            missing = cleaned_df.isna().sum().sum()
            if missing > 0:
                print(f"Warning: {missing} missing values remain in {attr_name}")

        return merged_data

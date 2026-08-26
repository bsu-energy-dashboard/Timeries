"""Load meter CSVs, build covariates (holidays, sessions), and fetch hourly weather."""

import json
import os
from importlib import resources
import datetime as datetime
from datetime import timedelta

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


def _default_holiday_dict() -> dict:
    """Load packaged BSU-style holiday ranges (JSON)."""
    data = resources.files("timeries.data").joinpath("holidaydict.json")
    with data.open("r", encoding="utf-8") as f:
        return json.load(f)


class DataProcessor:
    """Process hourly meter CSVs, optional virtual meters, covariates, and weather."""

    def __init__(
        self,
        file_path,
        start_date,
        end_date,
        addmeter=None,
        BreaksWeather=None,
        holiday_dict_path=None,
    ):
        if BreaksWeather is None:
            BreaksWeather = [True, True, True, True]
        self.file_path = file_path
        self._holiday_dict_path = holiday_dict_path
        self.holidaydict = self.__readholidaydict()
        self.start_date = self.__parse_date(start_date)
        self.end_date = self.__parse_date(end_date)

        if self.start_date > self.end_date:
            raise ValueError("Start date must be before or at the end date")

        self.addmeter = addmeter
        self.dataframe = self.__process()

        self.__BreaksWeather = BreaksWeather
        self.covariates = None
        self.__createcovariates()

        if self.__BreaksWeather[3]:
            self.weather_dataframe = self.__makeweather_data()

    def _cache_dir(self) -> str:
        base = os.path.dirname(os.path.abspath(self.file_path))
        path = os.path.join(base, ".cache")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def __parse_date(date_text: str) -> datetime.datetime:
        try:
            return pd.to_datetime(date_text)
        except ValueError:
            raise ValueError(f"Invalid date format: {date_text}. Expected YYYY-MM-DD.")

    def __process(self):
        date_column = "Date / Time"
        datain = pd.read_csv(self.file_path, index_col=date_column, parse_dates=True)
        datain.index = pd.to_datetime(datain.index, errors="coerce")
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)

        dataframe = datain.loc[start_dt:end_dt].copy(deep=True)

        if dataframe.empty:
            raise ValueError(f"DataFrame empty. Range: {datain.index.min()} to {datain.index.max()}")

        if self.addmeter is not None:
            for value in self.addmeter.values():
                if not set(value).issubset(set(dataframe.columns)):
                    raise ValueError(f"Columns {value} must be in the csv.")
            for meter, values in self.addmeter.items():
                dataframe.loc[:, meter] = dataframe.loc[:, values].sum(axis=1)

        return dataframe

    def __readholidaydict(self):
        if self._holiday_dict_path is not None:
            with open(self._holiday_dict_path, "r", encoding="utf-8") as f:
                return json.load(f)
        local_upper = os.path.join(os.path.dirname(self.file_path), "HOLIDAYDICT.JSON")
        local_lower = os.path.join(os.path.dirname(self.file_path), "HOLIDAYDICT.json")
        for path in (local_upper, local_lower):
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return _default_holiday_dict()

    def __BSU_holiday_add(self, df):
        print("Adding BSU Holidays")
        df["Holiday_name"] = "No Holiday"
        for holiday_name, date_range in list(self.holidaydict.items())[:-1]:
            for i in range(3):
                start_date = pd.to_datetime(date_range[2 * i])
                end_date = pd.to_datetime(date_range[2 * i + 1])
                findholiday = (df.index.date >= start_date.date()) & (df.index.date <= end_date.date())
                df.loc[findholiday, "Holiday_name"] = holiday_name

        df["IsNotHoliday"] = df["Holiday_name"] == "No Holiday"
        df["IsNotHoliday"] = df["IsNotHoliday"].astype(int)
        return df

    def __Summers_add(self, df):
        print("Adding Summer Break")
        SBD = pd.to_datetime(self.holidaydict["Summer Break"])

        is_summer_2023 = (df.index.date >= SBD[0].date()) & (df.index.date <= SBD[1].date())
        is_summer_2024 = (df.index.date >= SBD[2].date()) & (df.index.date <= SBD[3].date())
        is_summer_2025 = (df.index.date >= SBD[4].date()) & (df.index.date <= SBD[5].date())

        findsummerbreak = is_summer_2023 | is_summer_2024 | is_summer_2025
        df["IsnotSummerBreak"] = 1 - findsummerbreak.astype(int)
        return df

    def __Weekends_add(self, df):
        print("Adding Weekends")
        df["is_not_weekend"] = (df.index.dayofweek != 5) & (df.index.dayofweek != 6)
        df["is_not_weekend"] = df["is_not_weekend"].astype(int)
        return df

    def __makeweather_data(self):
        print("Making weather data")
        weather_forecast = self.__process_forecast()
        if self.dataframe.index.min() < datetime.datetime.now() - timedelta(days=60):
            pastweather = self.__process_past_weather()
            weatherdf = pd.concat([pastweather, weather_forecast])
        else:
            weatherdf = weather_forecast

        weatherdf = weatherdf.sort_index()

        if weatherdf.index.tz is not None:
            weatherdf.index = weatherdf.index.tz_localize(None)

        target_start = pd.to_datetime(self.start_date)
        new_index = pd.date_range(start=target_start, periods=len(weatherdf), freq="h")
        weatherdf.index = new_index

        return weatherdf

    def __createcovariates(self):
        start_ts = pd.to_datetime(self.start_date)
        end_ts = pd.to_datetime(self.end_date)
        date_index = pd.date_range(start=start_ts, end=end_ts, freq="h")
        covariates = pd.DataFrame(index=date_index)

        self.covariates = self.__Weekends_add(covariates)
        if self.__BreaksWeather[0]:
            self.covariates = self.__BSU_holiday_add(self.covariates)
        if self.__BreaksWeather[1]:
            self.covariates = self.__Summers_add(self.covariates)
        if self.__BreaksWeather[2]:
            self.covariates["InSession"] = (
                self.covariates["IsNotHoliday"]
                * self.covariates["is_not_weekend"]
                * self.covariates["IsnotSummerBreak"]
            )

    def __process_past_weather(self) -> pd.DataFrame:
        print("Processing past weather")
        cache_session = requests_cache.CachedSession(self._cache_dir(), expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        open_meteo = openmeteo_requests.Client(session=retry_session)
        current_time = datetime.datetime.now() - timedelta(days=61)
        TIMEZONE = "America/Indiana/Indianapolis"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 40.2035,
            "longitude": -85.4064,
            "hourly": ("temperature_2m"),
            "start_date": (self.dataframe.index.min()).strftime("%Y-%m-%d"),
            "end_date": current_time.strftime("%Y-%m-%d"),
            "timezone": TIMEZONE,
            "temperature_unit": "fahrenheit",
        }
        try:
            responses = open_meteo.weather_api(url, params=params)
            response = responses[0]
            hourly = response.Hourly()

            df_hourly = pd.DataFrame(
                {
                    "DateTime": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left",
                    ),
                    "Temperature (°F)": hourly.Variables(0).ValuesAsNumpy(),
                }
            )
            df_hourly = df_hourly.set_index("DateTime")
            return df_hourly
        except Exception as e:
            print(f"Error processing past data: {e}")
        return pd.DataFrame()

    def __process_forecast(self) -> pd.DataFrame:
        cache_session = requests_cache.CachedSession(self._cache_dir(), expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        open_meteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 40.2035,
            "longitude": -85.4064,
            "hourly": "temperature_2m",
            "timezone": "America/Indiana/Indianapolis",
            "forecast_days": 10,
            "temperature_unit": "fahrenheit",
            "past_days": 60,
        }
        try:
            responses = open_meteo.weather_api(url, params=params)
            response = responses[0]
            hourly = response.Hourly()

            df_hourly = pd.DataFrame(
                {
                    "DateTime": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left",
                    ),
                    "Temperature (°F)": hourly.Variables(0).ValuesAsNumpy(),
                }
            )
            df_hourly = df_hourly.set_index("DateTime")
            return df_hourly
        except Exception as e:
            print(f"Error processing forecast data: {e}")
            return pd.DataFrame()

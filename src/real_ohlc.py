import pandas as pd
import numpy as np
from constants.constants import *


class RealOHLC:
    def __init__(self, data_choice: str, num_days: int) -> None:
        self.total_days = num_days
        self.data_choice = data_choice
        self.df = None
        self.resampled_data = {}
        self.agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

    def create_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create a dataframe for real data"""

        df = pd.read_csv(
            self.data_choice,
            usecols=["date", "open", "high", "low", "close"],
            skiprows=1,
        )[::-1]

        df = df.drop(df[df["date"] < start_date].index)
        df = df.drop(df[df["date"] > end_date].index)

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.set_index("date", inplace=True)
        self.df = df

    def normalize_ohlc_data(self) -> pd.DataFrame:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """

        _max = np.max(
            [
                np.max(self.df.open),
                np.max(self.df.high),
                np.max(self.df.low),
                np.max(self.df.close),
            ]
        )
        _min = np.min(
            [
                np.min(self.df.open),
                np.min(self.df.high),
                np.min(self.df.low),
                np.min(self.df.close),
            ]
        )

        norm_open = (self.df.open - _min) / (_max - _min)
        norm_high = (self.df.high - _min) / (_max - _min)
        norm_low = (self.df.low - _min) / (_max - _min)
        norm_close = (self.df.close - _min) / (_max - _min)

        random_multiplier = np.random.randint(9, 999)

        self.df["open"] = np.round(norm_open * random_multiplier, 4)
        self.df["high"] = np.round(norm_high * random_multiplier, 4)
        self.df["low"] = np.round(norm_low * random_multiplier, 4)
        self.df["close"] = np.round(norm_close * random_multiplier, 4)

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        prev_timeframe = "1min"
        self.resampled_data["1min"] = self.df
        bars_table = self.__create_bars_table()

        for timeframe in bars_table:
            self.resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.resampled_data[prev_timeframe]
            )
            prev_timeframe = timeframe

    def __downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Converts a higher resolution dataframe into a lower one.

        For example:
            converts 1min candle sticks into 5min candle sticks.

        The closed parameter controls which end of the interval is inclusive
        while the label parameter controls which end of the interval appears on the resulting index.
        right and left refer to end and the start of the interval, respectively.
        """
        return df.resample(timeframe, label="right", closed="right").aggregate(
            self.agg_dict
        )

    def __create_bars_table(self) -> dict:
        return {
            "1min": self.total_days * MINUTES_IN_1DAY,
            "5min": self.total_days * MINUTES_IN_1DAY // 5,
            "15min": self.total_days * MINUTES_IN_1DAY // 15,
            "30min": self.total_days * MINUTES_IN_1DAY // 30,
            "1H": self.total_days * HOURS_IN_1DAY,
            "2H": self.total_days * HOURS_IN_1DAY // 2,
            "4H": self.total_days * HOURS_IN_1DAY // 4,
            "1D": self.total_days,
            "3D": self.total_days // 3,
            "1W": self.total_days // 7,
            "1M": self.total_days // 30,
        }

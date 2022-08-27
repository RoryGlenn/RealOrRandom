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
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }

    def create_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create a dataframe for real data"""
        df = pd.read_csv(
            self.data_choice,
            usecols=["Date", "Open", "High", "Low", "Close"],
            skiprows=1,
        )[::-1]

        df = df.drop(df[df["Date"] < start_date].index)
        df = df.drop(df[df["Date"] > end_date].index)

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.set_index("Date", inplace=True)
        self.df = df

    def normalize_ohlc_data(self) -> pd.DataFrame:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """

        _max = np.max(
            [
                np.max(self.df.Open),
                np.max(self.df.High),
                np.max(self.df.Low),
                np.max(self.df.Close),
            ]
        )
        _min = np.min(
            [
                np.min(self.df.Open),
                np.min(self.df.High),
                np.min(self.df.Low),
                np.min(self.df.Close),
            ]
        )

        norm_open = (self.df.Open - _min) / (_max - _min)
        norm_high = (self.df.High - _min) / (_max - _min)
        norm_low = (self.df.Low - _min) / (_max - _min)
        norm_close = (self.df.Close - _min) / (_max - _min)

        random_multiplier = np.random.randint(9, 999)

        self.df["Open"] = np.round(norm_open * random_multiplier, 4)
        self.df["High"] = np.round(norm_high * random_multiplier, 4)
        self.df["Low"] = np.round(norm_low * random_multiplier, 4)
        self.df["Close"] = np.round(norm_close * random_multiplier, 4)

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

    def abstract_dates(self) -> None:
        """Remove the real dates and replace them with fake dates"""

        self.df.reset_index(inplace=True)
        date = "date" if "date" in self.df.columns else "Date"

        dates_new = pd.DataFrame(
            {
                date: np.tile(
                    pd.date_range(
                        start="2000-01-01",
                        periods=len(self.df),
                        freq="1min",
                    ),
                    1,
                ),
            }
        )

        self.df[date] = dates_new[date]
        self.df.set_index(date, inplace=True)

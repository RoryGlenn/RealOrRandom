from datetime import date
from statistics import stdev
from time import perf_counter

import random
from timeit import timeit
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from faker import Faker

from constants.constants import *

pd.options.display.float_format = "{:.4f}".format


class RandomOHLC:
    def __init__(
        self,
        total_days: int,
        start_price: float,
        name: str,
        volatility: int,
    ) -> None:

        self.total_days = total_days
        self.start_price = start_price
        self.name = name
        self.volatility = volatility
        self.__distribution_functions = {
            1: np.random.normal,
            2: np.random.laplace,
            3: np.random.logistic,
            # 4: brownian,
        }

        self.__df: pd.DataFrame
        self.__resampled_data = {
            # Put these back in when testing is over!
            "1min": None,
            "5min": None,
            "15Min": None,
            "30Min": None,
            "1H": None,
            "2H": None,
            "4H": None,
            "1D": None,
            "3D": None,
            "1W": None,
            "1M": None,
        }

        self.sd = {"open": 0, "high": 0, "low": 0, "close": 0}
        self.vp_random_chance = None

    @property
    def resampled_data(self) -> dict:
        return self.__resampled_data

    @staticmethod
    def get_time_elapsed(start_time: float) -> float:
        return round(perf_counter() - start_time, 2)

    def __set_sd_ohlc(self) -> list:
        """Sets the standard deviation of the ohlc data"""
        self.sd["open"] = stdev(self.__df["open"])
        self.sd["high"] = stdev(self.__df["high"])
        self.sd["low"] = stdev(self.__df["low"])
        self.sd["close"] = stdev(self.__df["close"])

    def generate_random_date(self) -> str:
        return Faker().date_between(
            start_date=date(year=1990, month=1, day=1), end_date="+1y"
        )

    def __brownian_motion(self, rows: int) -> np.ndarray:
        """Creates a brownian motion ndarray"""
        T = 1.0
        dimensions = 1
        times = np.linspace(0.0, T, rows)
        dt = times[1] - times[0]

        # Bt2 - Bt1 ~ Normal with mean 0 and variance t2-t1
        dB = np.sqrt(dt) * np.random.normal(size=(rows - 1, dimensions))
        B0 = np.zeros(shape=(1, dimensions))
        B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
        return B

    def __brownian_motion_distribution(self, rows: int) -> list[float]:
        """Returns a dataframe with a brownian motion distribution"""
        bm_array = self.__brownian_motion(rows)
        bm_array = [i[0] for i in bm_array]
        return self.__normalize_ohlc_list(bm_array)

    def normalize_ohlc_data(self) -> None:
        """
        Normalize OHLC data with random multiplier.
        normalization formula: (data - min) / (max - min)
        """

        print("Normalizing OHLC data..")

        # FIND THE BOTTLE NECK IN THIS FUNCTION

        start1 = perf_counter()
        _max = np.max(
            [
                np.max(self.__df.open),
                np.max(self.__df.high),
                np.max(self.__df.low),
                np.max(self.__df.close),
            ]
        )

        _min = np.min(
            [
                np.min(self.__df.open),
                np.min(self.__df.high),
                np.min(self.__df.low),
                np.min(self.__df.close),
            ]
        )

        norm_open = (self.__df.open - _min) / (_max - _min)
        norm_high = (self.__df.high - _min) / (_max - _min)
        norm_low = (self.__df.low - _min) / (_max - _min)
        norm_close = (self.__df.close - _min) / (_max - _min)

        random_multiplier = np.random.randint(9, 999)

        self.__df["open"] = round(norm_open * random_multiplier, 4)
        self.__df["high"] = round(norm_high * random_multiplier, 4)
        self.__df["low"] = round(norm_low * random_multiplier, 4)
        self.__df["close"] = round(norm_close * random_multiplier, 4)

    def __create_dataframe(self, num_bars: int, frequency: str, prices: np.ndarray, start_date: str = '2000-01-01'):
        return pd.DataFrame(
            {
                "date": np.tile(
                    pd.date_range(
                        start=start_date,
                        periods=num_bars,
                        freq=frequency,
                    ),
                    1,
                ),
                "price": (prices),
            }
        )

    def generate_random_df(
        self,
        num_bars: int,
        frequency: str,
        start_price: float,
        volatility: int,
    ) -> pd.DataFrame:
        self.__df = self.__generate_random_df(
            num_bars, frequency, start_price, volatility
        )

    def __generate_random_df(
        self,
        num_bars: int,
        frequency: str,
        start_price: float,
        volatility: int,
    ) -> pd.DataFrame:

        # start1 = perf_counter()
        dist_func = self.__distribution_functions.get(
            np.random.randint(1, len(self.__distribution_functions))
        )
        steps = dist_func(scale=volatility, size=num_bars)
        steps[0] = 0
        prices = start_price + np.cumsum(steps)
        prices = np.abs(prices.round(decimals=6))
        # print("start1: ", perf_counter() - start1)

        # start2 = perf_counter()
        #########################################
        # BOTTLENECK
        df = self.__create_dataframe(num_bars, frequency, prices)
        #########################################
        # print("start2: ", perf_counter() - start2)

        # start3 = perf_counter()
        df.index = pd.to_datetime(df.date)
        df = df.price.resample("1min").ohlc(_method="ohlc")
        # print("start3: ", perf_counter() - start3)
        return df

    def set_get_vpchance(self, random_chance: int = None) -> int:
        if self.vp_random_chance is None:
            self.vp_random_chance = random_chance
        return self.vp_random_chance

    def __create_volatile_periods(self) -> None:
        """Create more volatile periods and replace the normal ones with them"""
        print("Creating volatile periods...")

        random_chance = np.random.randint(1, 100)
        indices = {}
        # total_1 = 0
        # total_2 = 0
        # start_1 = perf_counter()

        for i in range(len(self.__df)):
            if np.random.randint(1, VOLATILE_PROB) < random_chance:
                vol_period_num_bars = np.random.randint(
                    VOLATILE_PERIOD_MIN, VOLATILE_PERIOD_MAX
                )

                if vol_period_num_bars + i < len(self.__df):
                    indices[i] = i + vol_period_num_bars

        for starti, endi in tqdm(indices.items()):
            # start_1 = perf_counter()

            new_df = self.__generate_random_df(
                num_bars=endi - starti + 1,
                frequency="1Min",
                start_price=self.__df.iloc[starti]["close"],
                volatility=self.volatility * np.random.uniform(1.01, 1.02),
            )
            # total_1 += perf_counter() - start_1

            # start_2 = perf_counter()
            self.__df.loc[starti:endi, "open"] = new_df["open"].values
            self.__df.loc[starti:endi, "high"] = new_df["high"].values
            self.__df.loc[starti:endi, "low"] = new_df["low"].values
            self.__df.loc[starti:endi, "close"] = new_df["close"].values
            # total_2 += perf_counter() - start_2
        # print(total_1, total_2)

    def __connect_open_close_candles(self) -> None:
        """Returns a dataframe where every candles close is the next candles open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.
        """

        print("Connecting open and closing candles...")

        prev_close = self.__df["close"].shift(1).fillna(0).astype(float)
        self.__df["open"] = prev_close
        self.__df.at[0, "open"] = self.start_price

        # LOW ERROR
        # get all the rows where the 'low' cell is not the lowest value
        self.__df["low"] = np.where(
            (
                (self.__df["low"] > self.__df["open"])
                | (self.__df["low"] > self.__df["high"])
                | (self.__df["low"] > self.__df["close"])
            ),
            # fix the low error by assigning this formula
            self.__df["low"]
            - abs(
                min(
                    np.min(self.__df["open"]),
                    np.min(self.__df["high"]),
                    np.min(self.__df["close"]),
                )
                - self.__df["low"]
            ),
            # assign the original value if no error occurred at the current index
            self.__df["low"],
        )

        # HIGH ERROR
        # get all the rows where the 'high' cell is not the highest value
        self.__df["high"] = np.where(
            (
                (self.__df["high"] < self.__df["open"])
                | (self.__df["high"] < self.__df["low"])
                | (self.__df["high"] < self.__df["close"])
            ),
            # fix the low error by assigning this formula
            self.__df["high"]
            + abs(
                max(
                    np.min(self.__df["open"]),
                    np.min(self.__df["low"]),
                    np.min(self.__df["close"]),
                )
                - self.__df["high"]
            ),
            # assign the original value if no error occurred at the current index
            self.__df["high"],
        )

        ##########################################################################

    def create_realistic_ohlc(self) -> None:
        """Process for creating slightly more realistic candles"""
        self.__df.reset_index(inplace=True)
        self.__create_volatile_periods()
        self.__connect_open_close_candles()
        self.__df.set_index("date", inplace=True)

    def __downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Converts a higher resolution dataframe into a lower one.

        For example:
            converts 1min candle sticks into 5min candle sticks.
        """
        return df.resample(timeframe).aggregate(
            {
                "open": lambda s: s[0],
                "high": lambda df: df.max(),
                "low": lambda df: df.min(),
                "close": lambda df: df[-1],
            }
        )

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        for timeframe in tqdm(self.__resampled_data):
            # since we already resampled self.df_post to 1min,
            # there is no need to resample it again.
            if timeframe == "1min":
                # is the 1min df stored in resampled data?
                self.__resampled_data[timeframe] = self.__df
                continue

            # resample the same df each time
            self.__resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.__df
            )

    def drop_dates(self) -> None:
        """Drops the date column on all dataframes"""
        print("Dropping date columns...")

        for timeframe, df in self.__resampled_data.items():
            self.__resampled_data[timeframe].reset_index(inplace=True)
            self.__resampled_data[timeframe].drop(columns=["date"], inplace=True)

    def print_resampled_data(self) -> None:
        {print(tf + "\n", df) for tf, df in self.resampled_data.items()}

from datetime import date
from time import perf_counter

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

        # Use Statistical functions (scipy.stats) instead of these!
        self.__distribution_functions = {
            1: np.random.normal,
            2: np.random.laplace,
            3: np.random.logistic,
            # 4: brownian,
            # np.random.random_sample()???
        }

        self.__df_1min: pd.DataFrame = None
        self.__resampled_data = {
            "1min": None,
            "5min": None,
            "15min": None,
            "30min": None,
            "1H": None,
            "2H": None,
            "4H": None,
            "1D": None,
            "3D": None,
            "1W": None,
            "1M": None,
        }
        self.agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }

    @property
    def resampled_data(self) -> dict:
        return self.__resampled_data

    @staticmethod
    def get_time_elapsed(start_time: float) -> float:
        return round(perf_counter() - start_time, 2)

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

        _max = np.max(
            [
                np.max(self.__df_1min.Open),
                np.max(self.__df_1min.High),
                np.max(self.__df_1min.Low),
                np.max(self.__df_1min.Close),
            ]
        )
        _min = np.min(
            [
                np.min(self.__df_1min.Open),
                np.min(self.__df_1min.High),
                np.min(self.__df_1min.Low),
                np.min(self.__df_1min.Close),
            ]
        )
        norm_open = (self.__df_1min.Open - _min) / (_max - _min)
        norm_high = (self.__df_1min.High - _min) / (_max - _min)
        norm_low = (self.__df_1min.Low - _min) / (_max - _min)
        norm_close = (self.__df_1min.Close - _min) / (_max - _min)

        random_multiplier = np.random.randint(9, 999)
        self.__df_1min.Open = round(norm_open * random_multiplier, 4)
        self.__df_1min.High = round(norm_high * random_multiplier, 4)
        self.__df_1min.Low = round(norm_low * random_multiplier, 4)
        self.__df_1min.Close = round(norm_close * random_multiplier, 4)

    def __create_dataframe(
        self,
        num_bars: int,
        frequency: str,
        prices: np.ndarray,
        start_date: str = "2000-01-01",
    ):
        return pd.DataFrame(
            {
                "Date": np.tile(
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
        self.__df_1min = self.__generate_random_df(
            num_bars, frequency, start_price, volatility
        )

    def __generate_random_df(
        self,
        num_bars: int,
        frequency: str,
        start_price: float,
        volatility: int,
    ) -> pd.DataFrame:
        dist_func = self.__distribution_functions.get(
            np.random.randint(1, len(self.__distribution_functions))
        )
        steps = dist_func(scale=volatility, size=num_bars)
        steps[0] = 0
        prices = start_price + np.cumsum(steps)
        prices = np.abs(prices.round(decimals=6))

        df = self.__create_dataframe(num_bars, frequency, prices)
        df.set_index("Date", inplace=True)

        # THE BIGGEST BOTTLE NECK IS HERE!!!
        ##################################
        df = df["price"].resample("1min", label="right", closed="right").ohlc()
        # df = df["price"].resample("1min", label="left", closed="left").ohlc()
        ##################################

        df.reset_index(inplace=True)
        df.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
            inplace=True,
        )
        return df

    def __create_volatile_periods(self) -> None:
        """Create more volatile periods and replace the normal ones with them"""
        print("Creating volatile periods...")

        length = len(self.__df_1min)
        random_chance = np.random.randint(1, 100)
        start_indices = np.arange(length)

        # Instead of a conditional statement, create a mask
        mask = np.random.randint(1, high=VOLATILE_PROB, size=length) < random_chance

        # make vol_period_num_bars an array
        array_masked = mask * np.random.randint(
            VOLATILE_PERIOD_MIN, high=VOLATILE_PERIOD_MAX, size=length
        )

        # Instead of the second "if" statement, exclude more values from the mask
        mask *= (start_indices + array_masked) < length
        start_indices = mask * (start_indices + array_masked)

        # For each index in indices that is not equal to 0 (aka start index), generate another index (aka end index)
        # such that it is larger than the current index but lower than 2*VOLATILE_PERIOD_MAX.
        # Attach this index as a key:value relationship in indices.
        end_indices = start_indices + (
            mask
            * np.random.randint(
                VOLATILE_PERIOD_MIN, high=VOLATILE_PERIOD_MAX, size=length
            )
        )

        # drop all values that are 0
        start_indices = start_indices[start_indices > 0]
        end_indices = end_indices[end_indices > 0]

        # fixes any out of bounds issues with the end index being greater than the length of the df
        end_indices = np.where(
            end_indices > len(self.__df_1min) - 1, len(self.__df_1min) - 1, end_indices
        )

        # For the second loop, you need to rewrite generate_random_df() to take an array as an argument.
        # Then you can get rid of the loop.
        for starti, endi in zip(tqdm(start_indices), end_indices):
            new_df = self.__generate_random_df(
                num_bars=endi - starti + 1,
                frequency="1min",
                start_price=self.__df_1min.iloc[starti]["Close"],
                volatility=self.volatility * np.random.uniform(1.01, 1.02),
            )

            self.__df_1min.loc[starti:endi, "Open"] = new_df["Open"].values
            self.__df_1min.loc[starti:endi, "High"] = new_df["High"].values
            self.__df_1min.loc[starti:endi, "Low"] = new_df["Low"].values
            self.__df_1min.loc[starti:endi, "Close"] = new_df["Close"].values

    def __correct_lowcolumn_error(self, df: pd.DataFrame) -> np.ndarray:
        """get all the rows where the 'Low' cell is not the lowest value"""

        conditions = [
            # figure out which value is the lowest value and assign it to the Low column
            (df["Open"] < df["High"]) & (df["Open"] < df["Close"]),
            (df["High"] < df["Open"]) & (df["High"] < df["Close"]),
            (df["Close"] < df["High"]) & (df["Close"] < df["Open"]),
        ]
        choices = [df["Open"], df["High"], df["Close"]]
        return np.where(
            (
                (df["Low"] > df["Open"])
                | (df["Low"] > df["High"])
                | (df["Low"] > df["Close"])
            ),
            # assign the minimum value
            np.select(conditions, choices),
            # assign the original value if no error occurred at the current index
            df["Low"],
        )

    def __correct_highcolumn_error(self, df: pd.DataFrame) -> np.ndarray:
        """Get all the rows where the 'High' cell is not the highest value"""

        conditions = [
            # figure out which value is the highest value and assign it to the High column
            (df["Open"] > df["Low"]) & (df["Open"] > df["Close"]),
            (df["Low"] > df["Open"]) & (df["Low"] > df["Close"]),
            (df["Close"] > df["Open"]) & (df["Close"] > df["Low"]),
        ]

        choices = [df["Open"], df["Low"], df["Close"]]

        return np.where(
            (
                (df["High"] < df["Open"])
                | (df["High"] < df["Low"])
                | (df["High"] < df["Close"])
            ),
            # assign the maximum value
            np.select(conditions, choices),
            # assign the original value if no error occurred at the current index
            df["High"],
        )

    def __connect_open_close_candles(self) -> None:
        """Returns a dataframe where every candles Close is the next candles Open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.

        """
        prev_close = self.__df_1min["Close"].shift(1).fillna(0).astype(float)
        self.__df_1min["Open"] = prev_close

        self.__df_1min.at[1, "Open"] = self.start_price
        self.__df_1min.drop([0], axis=0, inplace=True)

        self.__df_1min.Low = self.__correct_lowcolumn_error(self.__df_1min)
        self.__df_1min.High = self.__correct_highcolumn_error(self.__df_1min)

    def create_realistic_ohlc(self) -> None:
        """Process for creating slightly more realistic candles"""
        # self.__create_volatile_periods()
        self.__connect_open_close_candles()
        self.__df_1min.set_index("Date", inplace=True)

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

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        prev_timeframe = "1min"
        self.__resampled_data["1min"] = self.__df_1min
        bars_table = self.__create_bars_table()

        for timeframe in bars_table:
            self.__resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.__resampled_data[prev_timeframe]
            )
            prev_timeframe = timeframe

    def print_resampled_data(self) -> None:
        {print(tf + "\n", df) for tf, df in self.resampled_data.items()}

import random
import numpy as np
import pandas as pd

from constants.constants import *


class RandomOHLC:
    def __init__(
        self, num_days_range: int, start_price: float, start_date: str, name: str
    ) -> None:
        self.num_days_range = num_days_range
        self.start_price = start_price
        self.start_date = start_date
        self.name = name

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

    def __brownian_motion_distribution(self, rows: int = 2880) -> list[float]:
        """Returns a dataframe with a brownian motion distribution"""
        bm_array = self.__brownian_motion(rows)
        bm_array = [i[0] for i in bm_array]
        return self.__normalize_ohlc_list(bm_array)

    def __normalize_ohlc_list(self, data: list) -> list:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """
        random_multiplier = random.randint(9, 999)

        _max = np.max(data)
        _min = np.min(data)

        norm_data: np.ndarray = ((data - _min) / (_max - _min)).tolist()
        return [round(i * random_multiplier, 4) for i in norm_data]

    # def normalize_ohlc_data(self, open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    #                         ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    #     """Normalize OHLC data with random multiplier
    #         normalization formula: (data - min) / (max - min)
    #     """

    #     _max = np.max([np.max(open), np.max(high), np.max(low), np.max(close)])
    #     _min = np.min([np.min(open), np.min(high), np.min(low), np.min(close)])

    #     norm_open = (open - _min) / (_max - _min)
    #     norm_high = (high - _min) / (_max - _min)
    #     norm_low = (low - _min) / (_max - _min)
    #     norm_close = (close - _min) / (_max - _min)

    #     random_multiplier = random.randint(9, 999)

    #     o = round(norm_open*random_multiplier, 4)
    #     h = round(norm_high*random_multiplier, 4)
    #     l = round(norm_low*random_multiplier, 4)
    #     c = round(norm_close*random_multiplier, 4)
    #     return o, h, l, c

    def __generate_random_crypto_df(
        self,
        days: int,
        start_price: float,
        col_name: str,
        start_date: str,
        volatility: int,
    ) -> pd.DataFrame:
        """Need to generate 1min, 5min, 15min, 1hr, 4hr, 1day, 1week"""

        distribution_dict = {
            1: np.random.normal,
            2: np.random.laplace,
            3: np.random.logistic,
        }

        # periods = days * 1440 # minutes
        periods = days * 24  # hours
        # periods = days        # days
        # periods = days / 7    # weeks ?

        # randomly pick a distribution function
        dist_func = distribution_dict.get(random.randint(1, 3))
        steps = dist_func(loc=0, scale=volatility, size=periods)
        steps[0] = 0

        price = start_price + np.cumsum(steps)
        price = [round(abs(p), 6) for p in price]

        return pd.DataFrame(
            {
                "ticker": np.repeat([col_name], periods),
                # <------ TEST THIS WITH 'D' INSTEAD OF 'H' <------
                "date": np.tile(
                    pd.date_range(self.start_date, periods=periods, freq="H"), 1
                ),
                "price": (price),
            }
        )

    def __downsample_ohlc_data(self, df: pd.DataFrame, timeframe: str) -> None:
        """Converts a higher resolution dataframe into a lower one.
        For example: converts 1min candle sticks into 5min candle sticks.

        # A helpful hint
        # https://towardsdatascience.com/pandas-resample-tricks-you-should-know-for-manipulating-time-series-data-7e9643a7e7f3

        TODO:
            Resample time frame to 1min, 5min, 15min, 1hr, 4hr, 1d, 1w
        """

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        df = df.resample("W").aggregate(
            {
                "open": lambda s: s[0],
                "high": lambda df: df.max(),
                "low": lambda df: df.min(),
                "close": lambda df: df[-1],
            }
        )
        return df

    def __create_whale_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a modified df containing whale values.
        Iterates over the dataframe and extends all values
        of the candle given a random chance.

        A single graph will have a random chance until a new graph is created.
        If the random number chosen is less than or equal to random_chance, create a whale candle.

        whale_mult:
            assigns a random floating point multiplier between the range of [WHALE_LOWER_MULT, WHALE_UPPER_MULT].
            By doing this, every time a whale candle is created, the probability
            of it being stretched with the same ratio as the
            previous whale candle or even the next whale candle is essentially 0%
        """

        # probability for creating a whale candle will be from 1-20%
        random_chance = random.randint(1, 20)

        df.reset_index(inplace=True)

        for i in range(len(df)):
            if random.randint(1, 100) <= random_chance:
                # assigns a random floating point multiplier between the range of [WHALE_LOWER_MULT, WHALE_UPPER_MULT].
                # By doing this, every time a whale candle is created, the probability
                # of it being stretched with the same ratio as the
                # previous whale candle or even the next whale candle is essentially 0%
                whale_mult = random.uniform(WHALE_LOWER_MULT, WHALE_UPPER_MULT)

                df.at[i, "open"] = df.iloc[i]["open"] * whale_mult
                df.at[i, "high"] = df.iloc[i]["high"] * whale_mult
                df.at[i, "low"] = df.iloc[i]["low"] / whale_mult
                df.at[i, "close"] = df.iloc[i]["close"] / whale_mult

        df.set_index("date", inplace=True)
        return df

    def __extend_wicks(self, df: pd.DataFrame, hl_mult: float) -> pd.DataFrame:
        """Returns a dataframe with the high and low wicks multiplied by the passed in hl_mult"""
        df.reset_index(inplace=True)
        for i in range(len(df)):
            new_h = df.iloc[i]["high"] * hl_mult
            new_l = df.iloc[i]["low"] - (df.iloc[i]["low"] * (hl_mult - 1))

            df.at[i, "high"] = new_h
            df.at[i, "low"] = new_l
        df.set_index("date", inplace=True)
        return df

    def __extend_all_wicks_randomly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe with the highs and lows multiplied by a random float"""
        df.reset_index(inplace=True)
        for i in range(len(df)):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            new_h = df.iloc[i]["high"] * h_mult
            new_l = df.iloc[i]["low"] - (df.iloc[i]["low"] * (l_mult - 1))

            df.at[i, "high"] = new_h
            df.at[i, "low"] = new_l
        df.set_index("date", inplace=True)
        df = self.__extend_wicks_randomly(df)
        return df

    def __extend_wicks_randomly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe with the highs, lows multiplied by a random float

        3 possibilities:
            extend only the high
            extend only the low
            extend both
        """

        df.reset_index(inplace=True)
        for i in range(len(df)):
            h_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)
            l_mult = random.uniform(RANDOM_LOWER_LIMIT, RANDOM_UPPER_LIMIT)

            random_choice = random.randint(1, 3)

            if random_choice == 1:
                # extend only the high
                df.at[i, "high"] = df.iloc[i]["high"] * h_mult
            elif random_choice == 2:
                # extend only the low
                df.at[i, "low"] = df.iloc[i]["low"] - (df.iloc[i]["low"] * (l_mult - 1))
            else:
                # extend both
                df.at[i, "high"] = df.iloc[i]["high"] * h_mult
                df.at[i, "low"] = df.iloc[i]["low"] - (df.iloc[i]["low"] * (l_mult - 1))
        df.set_index("date", inplace=True)
        return df

    def __connect_open_close_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a dataframe where every candles close is the next candles open.
        This is needed because cryptocurrencies run 24/7.
        There are no breaks or pauses so each candle is connected to the next candle.
        """
        df.reset_index(inplace=True)

        for i in range(1, len(df)):
            # connects each open and close together
            df.at[i, "open"] = df.iloc[i - 1]["close"]

            min_value = min(df.iloc[i]["open"], df.iloc[i]["high"], df.iloc[i]["close"])

            max_value = max(df.iloc[i]["open"], df.iloc[i]["low"], df.iloc[i]["close"])

            # something went wrong and the low is not the lowest value
            if df.iloc[i]["low"] > min_value:
                # get the difference between the low and the lowest value and subtract it from the low
                df.at[i, "low"] = df.iloc[i]["low"] - abs(min_value - df.iloc[i]["low"])

            # get the difference between the highest value and the high and add it to the high
            if df.iloc[i]["high"] < max_value:
                df.at[i, "high"] = df.iloc[i]["high"] + abs(
                    max_value - df.iloc[i]["high"]
                )

        df.set_index("date", inplace=True)
        return df

    def create_realistic_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        # make the candles look a little bit more real
        df = self.__create_whale_candles(df)
        df = self.__extend_all_wicks_randomly(df)
        return self.__connect_open_close_candles(df)

    def create_random_df(self) -> pd.DataFrame:
        """Create a dataframe for random data"""
        volatility = random.randint(100, 200)

        df = self.__generate_random_crypto_df(
            self.num_days_range,
            self.start_price,
            self.name,
            self.start_date,
            volatility=volatility,
        )

        df.index = pd.to_datetime(df.date)

        # 25% to use a brownian motion distribution instead
        if random.randint(1, 4) == 4:
            df["price"] = self.__brownian_motion_distribution()

        # HOW DOES this WORK?
        return df.price.resample("D").ohlc()

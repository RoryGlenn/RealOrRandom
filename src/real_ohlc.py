from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from constants.constants import MINUTES_IN_1DAY, HOURS_IN_1DAY, DATA_PATH, DATE_FORMAT


class RealOHLC:
    def __init__(
        self, data_choice: str, num_days: int, data_files: list[str] = []
    ) -> None:
        self.__num_days = num_days
        self.__data_choice = data_choice
        self.__data_files = data_files
        self.__resampled_data = {}
        self.__df = None
        self.__filename = None
        self.__start_date_dt = None
        self.__end_date_dt = None
        self.__start_date_str = None
        self.__end_date_str = None
        self.__agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }

    @property
    def data_choice(self) -> str:
        return self.__data_choice

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    @property
    def resampled_data(self) -> dict:
        return self.__resampled_data

    @property
    def start_date_dt(self) -> datetime:
        return self.__start_date_dt

    @property
    def end_date_dt(self) -> datetime:
        return self.__end_date_dt

    @property
    def start_date_str(self) -> str:
        return self.__start_date_str

    @property
    def end_date_str(self) -> str:
        return self.__end_date_str

    def get_filenames(self, path: str) -> list[str]:
        """open the folder containing all the .csv files and return a list containing all the files"""
        from os import listdir
        from os.path import isfile, join

        return [
            f
            for f in listdir(path)
            if isfile(join(path, f)) and join(path, f)[-4:] == ".csv"
        ]

    def set_file_choice(self) -> None:
        """randomly choose a file"""
        self.__filename = np.random.choice(self.__data_files)

    def set_start_end_datelimits(self) -> None:
        """Returns a dictionary containing all of the file names as the key
        and start/end dates as the value.
        The start date is also adjusted 90 days into the future
        to avoid out of bounds issues when a random start date is picked.
        """

        # protects loop against incorrectly placed files in the data folder
        if self.__filename not in self.__data_files:
            from sys import exit as sysexit

            print(f"{self.__filename} was not found in data repository")
            sysexit(1)

        df = pd.read_csv(DATA_PATH + "/" + self.__filename, skiprows=1)
        dt = datetime.strptime(df.loc[len(df) - 1, "Date"], DATE_FORMAT) + timedelta(
            days=90
        )
        self.__start_date_dt = dt.strftime(DATE_FORMAT)
        self.__end_date_dt = df.loc[0, "Date"]

    def randomly_pick_start_end_dates(self) -> None:
        """Once the start and end date limit have been set,
        Randomly select what our start date will be.
        Then add 'num_days' to the start_date, this will be the end_date."""

        start_date_dt = datetime.strptime(self.__start_date_dt, DATE_FORMAT)
        end_date_dt = datetime.strptime(self.__end_date_dt, DATE_FORMAT)

        # get the number of days from the start date to the end date
        diff_dt = end_date_dt - start_date_dt

        # Create a list of all the dates within the given date bounds.
        # Limit the total number of days we can use to diff_dt.days-num_days
        # so the last 'num_days' will be off limits to the start date.
        # By doing this, we protect ourselves from an out of bounds error
        dt_list = [
            start_date_dt + timedelta(days=x)
            for x in range(diff_dt.days - self.__num_days)
        ]

        if len(dt_list) == 0:
            from sys import exit as sysexit
            print("dt_list is empty!")
            sysexit(1)

        # # randomly choose a start date, then go 'num_days' into the future to get the end date
        start_date_dt = np.random.choice(dt_list)
        end_date_dt = start_date_dt + timedelta(days=self.__num_days)

        # create the start and end date strings
        self.__start_date_str = start_date_dt.strftime(DATE_FORMAT)
        self.__end_date_str = end_date_dt.strftime(DATE_FORMAT)

    def merge_csv_files(self, symbol_pair: str) -> pd.DataFrame:
        """Returns a merged dataframe given a symbol_pair.

        For all files in the data directory, look for the csv files containing
        the symbol_pair and combine them all into one dataframe.
        """
        files = [file for file in self.get_filenames(DATA_PATH) if symbol_pair in file]

        # set the first df
        df_master = pd.read_csv(
            DATA_PATH + "/" + files.pop(0),
            usecols=["Date", "Open", "High", "Low", "Close"],
            skiprows=1,
        )[::-1]

        # iterate through the remaining files and concat them to df_master
        for f in files:
            if symbol_pair in f:
                df = pd.read_csv(
                    DATA_PATH + "/" + f,
                    usecols=["Date", "Open", "High", "Low", "Close"],
                    skiprows=1,
                )[::-1]
                df_master = pd.concat([df_master, df], ignore_index=True)
        return df_master

    def create_df(self, merge_csvs: bool) -> pd.DataFrame:
        """Create a dataframe for real data"""
        df = None

        # if you already pick the start and end dates,
        # theres no reason to merge all of the csv files together

        if merge_csvs:
            symbol_pair = self.__data_choice.split("_")[1]
            df = self.merge_csv_files(symbol_pair)  # BOTTLE NECK HERE!
        else:
            df = pd.read_csv(
                self.__data_choice,
                usecols=["Date", "Open", "High", "Low", "Close"],
                skiprows=1,
            )[::-1]

        df = df.drop(df[df["Date"] < self.__start_date_str].index)
        df = df.drop(df[df["Date"] > self.__end_date_str].index)

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.set_index("Date", inplace=True)
        self.__df = df

    def normalize_ohlc_data(self) -> pd.DataFrame:
        """Normalize OHLC data with random multiplier
        normalization formula: (data - min) / (max - min)
        """

        _max = np.max(
            [
                np.max(self.__df.Open),
                np.max(self.__df.High),
                np.max(self.__df.Low),
                np.max(self.__df.Close),
            ]
        )
        _min = np.min(
            [
                np.min(self.__df.Open),
                np.min(self.__df.High),
                np.min(self.__df.Low),
                np.min(self.__df.Close),
            ]
        )

        norm_open = (self.__df.Open - _min) / (_max - _min)
        norm_high = (self.__df.High - _min) / (_max - _min)
        norm_low = (self.__df.Low - _min) / (_max - _min)
        norm_close = (self.__df.Close - _min) / (_max - _min)

        random_multiplier = np.random.randint(9, 999)

        self.__df["Open"] = np.round(norm_open * random_multiplier, 4)
        self.__df["High"] = np.round(norm_high * random_multiplier, 4)
        self.__df["Low"] = np.round(norm_low * random_multiplier, 4)
        self.__df["Close"] = np.round(norm_close * random_multiplier, 4)

    def resample_timeframes(self) -> None:
        """Iterates over all the timeframe keys in resampled_data and creates a
        resampled dataframe corresponding to that timeframe"""

        prev_timeframe = "1min"
        self.__resampled_data["1min"] = self.__df
        bars_table = self.__create_bars_table()

        for timeframe in bars_table:
            self.__resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.__resampled_data[prev_timeframe]
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
            self.__agg_dict
        )

    def __create_bars_table(self) -> dict:
        return {
            "1min": self.__num_days * MINUTES_IN_1DAY,
            "5min": self.__num_days * MINUTES_IN_1DAY // 5,
            "15min": self.__num_days * MINUTES_IN_1DAY // 15,
            "30min": self.__num_days * MINUTES_IN_1DAY // 30,
            "1H": self.__num_days * HOURS_IN_1DAY,
            "2H": self.__num_days * HOURS_IN_1DAY // 2,
            "4H": self.__num_days * HOURS_IN_1DAY // 4,
            "1D": self.__num_days,
            "3D": self.__num_days // 3,
            "1W": self.__num_days // 7,
            "1M": self.__num_days // 30,
        }

    def abstract_dates(self) -> None:
        """Remove the real dates and replace them with fake dates"""
        self.__df.reset_index(inplace=True)

        dates_new = pd.DataFrame(
            {
                "Date": np.tile(
                    pd.date_range(
                        start="2000-01-01",
                        periods=len(self.__df),
                        freq="1min",
                    ),
                    1,
                ),
            }
        )

        self.__df["Date"] = dates_new["Date"]
        self.__df.set_index("Date", inplace=True)

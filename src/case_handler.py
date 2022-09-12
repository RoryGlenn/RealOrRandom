import numpy as np
import pandas as pd
from sys import exit as sys_exit

# import cufflinks as cf
from faker import Faker

from dates import Dates
from download import Download
from real_ohlc import RealOHLC
from random_ohlc import RandomOHLC
from constants.constants import (
    DATA_PATH,
    DOWNLOAD_PATH,
    GITHUB_URL,
    SECONDS_IN_1DAY,
    DATA_FILENAMES,
    START_PRICE_RANDOM_CASE,
)


class CaseHandler:
    def __init__(self, num_days: int = 120) -> None:
        self.num_days = num_days
        self.dataframes = None
        self.half_dataframes = None
        self.results = {}
        self.answers = {}
        self.answer = None
        self.user_answers = {}
        self.faker = Faker()
        self.curr_graph_id = 0
        self.check_days = [1, 5, 10, 30, 60]

    def choose(self) -> bool:
        # return np.random.choice([True, False])
        return np.random.choice([True])

    def __create_half_dataframes(
        self, dataframes: dict[str, pd.DataFrame], exclusions=[]
    ) -> dict[str, pd.DataFrame]:
        """Creates a new dict that contains only the first half the data in the dataframes"""
        return {
            timeframe: df.iloc[: len(df) // 2]
            for timeframe, df in dataframes.items()
            if timeframe not in exclusions
        }

    def reset_indices(self) -> None:
        """Resets the index for every dataframe in dataframes and half_dataframes"""
        {
            df.reset_index(inplace=True): hdf.reset_index(inplace=True)
            for df, hdf in zip(self.dataframes.values(), self.half_dataframes.values())
        }

    def real_case(self, num_days: int, exclusions: list[str] = []) -> None:
        """Creates the real case scenario"""

        # get the files and randomly decide which data to use
        files = Dates.get_filenames(DATA_PATH)
        filename = np.random.choice(files)

        real_ohlc = RealOHLC(
            data_choice=filename,
            num_days=num_days,
            data_files=Download.get_data_filenames(DATA_FILENAMES),
        )

        real_ohlc.set_start_end_datelimits()
        real_ohlc.pick_start_end_dates()
        real_ohlc.create_df(merge_csvs=False)
        real_ohlc.normalize_ohlc_data()

        real_ohlc.abstract_dates()
        # real_ohlc.drop_first_day()

        real_ohlc.resample_timeframes()
        self.dataframes = real_ohlc.resampled_data.copy()

        if len(self.dataframes["1D"]) != self.num_days:
            print(
                f"len(self.dataframes['1D']): {len(self.dataframes['1D'])} != self.num_days: {self.num_days}"
            )
            # sys_exit(1)

        self.half_dataframes = self.__create_half_dataframes(
            real_ohlc.resampled_data.copy(), exclusions
        )

        self.answer = {
            "Real_Or_Random": "Real",
            "Name": self.faker.name(),
            "Start_Date": real_ohlc.start_date_str,
            "End_Date": real_ohlc.end_date_str,
            "File": real_ohlc.data_choice,
        }

    def random_case(self, num_days: int, exclusions: list[str] = []) -> None:
        random_ohlc = RandomOHLC(
            total_days=num_days,
            start_price=START_PRICE_RANDOM_CASE,
            name=self.faker.name(),
            volatility=np.random.uniform(1, 2),
        )
        random_ohlc.generate_random_df(
            random_ohlc.total_days * SECONDS_IN_1DAY,
            "1S",
            random_ohlc.start_price,
            random_ohlc.volatility,
        )
        random_ohlc.create_realistic_ohlc()
        random_ohlc.normalize_ohlc_data()

        print(self.dataframes)
        random_ohlc.resample_timeframes()
        self.dataframes = random_ohlc.resampled_data

        if len(self.dataframes["1D"]) != self.num_days:
            print(
                f"len(self.dataframes['1D']): {len(self.dataframes['1D'])} != self.num_days: {self.num_days}"
            )
            from sys import exit as sys_exit

            sys_exit(1)

        self.half_dataframes = self.__create_half_dataframes(
            random_ohlc.resampled_data, exclusions
        )
        self.answer = {
            "Real_Or_Random": "Random",
            "Name": self.faker.name(),
            "Start_Date": "None",
            "End_Date": "None",
            "File": "None",
        }

    @staticmethod
    def get_relative_change(initial_value: float, final_value: float) -> float:
        """Returns the relative change.
        Formula = (x2 - x1) / x1"""
        return (final_value - initial_value) / initial_value

    @staticmethod
    def get_results(
        users_answers: dict, relative_change: float, day_number: int
    ) -> dict:
        perror_perc = round(
            abs(relative_change) - abs(users_answers[f"{day_number}daybounds-slider"]),
            4,
        )
        prediction_perc = users_answers[f"{day_number}daybounds-slider"]
        prediction_rr = users_answers["realorrandom-dropdown"]
        pattern = users_answers["pattern-dropdown"]
        confidence = users_answers["confidence-slider"]

        try:
            d = {
                f"prediction": prediction_perc,
                f"prediction_error_percent": perror_perc,
                "prediction_real_or_random": prediction_rr,
                "pattern": pattern,
                "confidence": confidence,
            }
            return d
        except ValueError as ve:
            print(ve)
        return

    def calculate_results(self) -> None:
        """Compare the users guessed price to the actual price in the full dataframe"""
        # need to iterate over all graphs!!!!
        # right now, this only iterates over 1 graph
        from pprint import pprint

        for graph_id, usr_answr in self.user_answers.items():
            initial_index = len(self.half_dataframes["1D"]) - 1

            initial_price = self.half_dataframes["1D"].loc[
                len(self.half_dataframes["1D"]) - 1, "Close"
            ]

            try:
                future_prices = [
                    self.dataframes["1D"].loc[initial_index + t, "Close"]
                    for t in self.check_days
                ]
            except ValueError as ve:
                print(ve)
            except KeyError as ke:
                # why don't we generate 120 days? We are usually below this target amount
                print(ke)
            finally:
                print()

            relative_changes = [
                self.get_relative_change(initial_price, f_day) * 100
                for f_day in future_prices
            ]

            print("initial_index:", initial_index)
            print("initial_price:", initial_price)
            print("future_prices:", future_prices)
            print("relative_changes:", relative_changes)
            pprint(usr_answr)

            _result = {
                day_number: self.get_results(usr_answr, rel_chge, day_number)
                for rel_chge, day_number in zip(relative_changes, self.check_days)
            }
            pprint(_result)
            print()

    """

    If I can just redirect the user to the current page once the routine finishes, then the problem is solved.

    it seems that all state is stored in the url. 
    If this is indeed the case, you should have all state available in the callback that updates the page content already 
    (i assume that the url is an Input? If not so, you can just add it as a State to get the info).

    """

    def init(self) -> None:
        Faker.seed(np.random.randint(10_000))

        # Download.download_data(
        #     url=GITHUB_URL,
        #     files_to_download=Download.get_data_filenames(DATA_FILENAMES),
        #     download_path=DOWNLOAD_PATH,
        # )

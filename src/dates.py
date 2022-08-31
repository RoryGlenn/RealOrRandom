from datetime import datetime, timedelta

import numpy as np


class Dates:
    @staticmethod
    def get_filenames(path: str) -> list[str]:
        """open the folder containing all the .csv files and return a list containing all the files"""
        from os import listdir
        from os.path import isfile, join

        return [
            f
            for f in listdir(path)
            if isfile(join(path, f)) and join(path, f)[-4:] == ".csv"
        ]

    @staticmethod
    def create_dates(
        num_days_range: int, start_date_limit: str, end_date_limit: str
    ) -> tuple[str, str]:
        """Randomly pick a start and end date within the given starting and ending bounds"""

        start_date_limit_l = [int(i) for i in start_date_limit.split("-")]
        end_date_limit_l = [int(i) for i in end_date_limit.split("-")]

        start_limit_dt = datetime(
            year=start_date_limit_l[0],
            month=start_date_limit_l[1],
            day=start_date_limit_l[2],
        )

        end_limit_dt = datetime(
            year=end_date_limit_l[0], month=end_date_limit_l[1], day=end_date_limit_l[2]
        )

        # get the number of days from the start date to the end date
        date_range_limit = end_limit_dt - start_limit_dt

        # create a list of all the dates within the given date bounds
        dt_list = [
            start_limit_dt + timedelta(days=x) for x in range(date_range_limit.days)
        ]

        # pick a random day to start minus the given range
        start_i = np.random.randint(0, len(dt_list) - num_days_range)
        end_i = start_i + num_days_range

        start_random_dt = dt_list[start_i]
        end_random_dt = dt_list[end_i]
        return start_random_dt.strftime("%Y-%m-%d"), end_random_dt.strftime("%Y-%m-%d")

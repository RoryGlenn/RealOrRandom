import pandas as pd


class RealOHLC:
    def __init__(self, data_choice: str) -> None:
        self.data_choice = data_choice

    def real_case(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Create a dataframe for real data"""
        df = df.drop(df[df["date"] < start_date].index)
        df = df.drop(df[df["date"] > end_date].index)

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.set_index("date", inplace=True)
        return df

        # df = df_real.copy()
        # df = real_case(df, start_date, end_date)
        # answers[i] = f"Real: {start_date} to {end_date} {data_choice}"

    def create_df(self) -> pd.DataFrame:
        return pd.read_csv(
            self.data_choice,
            usecols=["date", "symbol", "open", "high", "low", "close"],
            skiprows=1,
        )[::-1]

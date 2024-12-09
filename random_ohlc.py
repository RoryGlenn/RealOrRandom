from datetime import date, datetime
from time import perf_counter
import numpy as np
import pandas as pd
from faker import Faker

pd.options.display.float_format = "{:.4f}".format


class RandomOHLC:
    def __init__(
        self,
        total_days: int,
        start_price: float,
        name: str,
        volatility: float,
        drift: float = 0.0,
    ) -> None:
        self.total_days = total_days
        self.start_price = start_price
        self.name = name
        self.volatility = volatility
        self.drift = drift

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

    def geometric_brownian_motion(
        self, start_price: float, num_steps: int, drift: float, volatility: float
    ) -> np.ndarray:
        """Simulate prices using Geometric Brownian Motion"""
        dt = 1 / num_steps
        prices = [start_price]
        for _ in range(num_steps - 1):
            shock = np.random.normal(0, 1) * np.sqrt(dt)
            prices.append(
                prices[-1]
                * np.exp((drift - 0.5 * volatility**2) * dt + volatility * shock)
            )
        return np.array(prices)

    def generate_random_df(
        self, num_bars: int, frequency: str, start_price: float, volatility: float
    ) -> pd.DataFrame:
        """Generate OHLC data using GBM"""
        prices = self.geometric_brownian_motion(
            start_price=start_price,
            num_steps=num_bars,
            drift=self.drift,
            volatility=volatility,
        )
        df = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=datetime.now(), periods=num_bars, freq=frequency
                ),
                "price": prices,
            }
        )
        df.set_index("date", inplace=True)
        ohlc = df["price"].resample("1min").ohlc()
        ohlc.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
            inplace=True,
        )
        return ohlc

    def create_realistic_ohlc(self, num_bars: int, frequency: str) -> None:
        """Generate realistic OHLC data"""
        self.__df_1min = self.generate_random_df(
            num_bars=num_bars,
            frequency=frequency,
            start_price=self.start_price,
            volatility=self.volatility,
        )
        self.__connect_open_close_candles()
        self.__add_volume_data()

    def __connect_open_close_candles(self) -> None:
        """Ensure consecutive candles are connected"""
        self.__df_1min["Open"] = (
            self.__df_1min["Close"].shift(1).fillna(self.start_price)
        )

    def __add_volume_data(self) -> None:
        """Add synthetic volume data based on volatility"""
        self.__df_1min["Volume"] = (
            np.abs(self.__df_1min["High"] - self.__df_1min["Low"])
            * np.random.randint(100, 1000, size=len(self.__df_1min))
        ).astype(int)

    def __downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLC data to lower timeframes"""
        return df.resample(timeframe).aggregate(self.agg_dict)

    def resample_timeframes(self) -> None:
        """Resample OHLC data into multiple timeframes"""
        self.__resampled_data["1min"] = self.__df_1min
        timeframes = ["5min", "15min", "30min", "1h", "4h", "1D"]
        for timeframe in timeframes:
            self.__resampled_data[timeframe] = self.__downsample_ohlc_data(
                timeframe, self.__resampled_data["1min"]
            )

from datetime import datetime
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

pd.options.display.float_format = "{:.4f}".format


class RandomOHLC:
    def __init__(
        self,
        total_days: int,
        start_price: float,
        name: str,
        volatility: float,
        drift: float,
    ) -> None:
        self.total_days = total_days
        self.start_price = start_price
        self.name = name
        self.volatility = volatility
        self.drift = drift

        self._df = None
        self._resampled_data = {
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
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

    @property
    def resampled_data(self) -> dict:
        return self._resampled_data

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

        # Converts a 2-dimensional time series dataframe into a 5-dimensional 
        # time series ohlc dataframe (open, high, low, close)
        ohlc = df["price"].resample(frequency).ohlc() 
        return ohlc

    def create_ohlc(self, num_bars: int, frequency: str) -> None:
        """Generate realistic OHLC data"""
        self._df = self.generate_random_df(
            num_bars=num_bars,
            frequency=frequency,
            start_price=self.start_price,
            volatility=self.volatility,
        )
        # self._connect_open_close_candles()

    def _connect_open_close_candles(self) -> None:
        """Ensure consecutive candles are connected"""
        self._df["open"] = self._df["close"].shift(1).fillna(self.start_price)


    def _downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLC data to lower timeframes"""
        return df.resample(timeframe).aggregate(self.agg_dict)

    def resample_timeframes(self) -> None:
        """Resample OHLC data into multiple timeframes"""
        self._resampled_data["1min"] = self._df
        timeframes = ["5min", "15min", "30min", "1h", "4h", "1D"]
        for timeframe in timeframes:
            self._resampled_data[timeframe] = self._downsample_ohlc_data(
                timeframe, self._resampled_data["1min"]
            )
        

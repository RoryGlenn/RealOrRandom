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
        drift: float = 0.0,
    ) -> None:
        self._total_days = total_days
        self._start_price = start_price
        self._name = name
        self._volatility = volatility
        self._drift = drift

        # self._df = None
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
        self._agg_dict = {
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
        self, num_bars: int, start_price: float, volatility: float
    ) -> pd.DataFrame:
        """Generate OHLC data using GBM"""
        
        num_bars *= 1440  # convert days to minutes

        prices = self.geometric_brownian_motion(
            start_price=start_price,
            num_steps=num_bars,
            drift=self._drift,
            volatility=volatility,
        )

        df = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=datetime.now(), periods=num_bars, freq='1min'
                ),
                "price": prices,
            }
        )
        df.set_index("date", inplace=True)
        
        # NOTE: basically copies and pastes the 1 minute price data into the open, high, low, close columns.
        # This is just to make the data look like OHLC data.
        ohlc_df = df["price"].resample("1min").ohlc()
        
        # connect the open prices to the close prices of the previous candle
        ohlc_df["open"] = self._connect_open_close_candles(ohlc_df)
        
        # resample the 1min data to 1D data and aggregate the open, high, low, close prices
        ohlc_df = ohlc_df.resample('1D').aggregate(self._agg_dict).round(2)

        return ohlc_df


    def _connect_open_close_candles(self, df) -> pd.DataFrame:
        """Ensure consecutive candles are connected"""
        return df["close"].shift(1).fillna(self._start_price)

    def _add_volume_data(self) -> None:
        """Add synthetic volume data based on volatility"""
        self._df["Volume"] = (
            self._df["high"]
            - self._df["low"]
            * np.random.randint(100, 1000, size=len(self._df))
        ).astype(int)

    def _downsample_ohlc_data(self, timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
        """Resample OHLC data to lower timeframes"""
        return df.resample(timeframe).aggregate(self._agg_dict)

    def _resample_timeframes(self, df) -> None:
        """Resample OHLC data into multiple timeframes"""
        self._resampled_data["1min"] = df
        timeframes = ["5min", "15min", "30min", "1h", "4h", "1D"]
        for timeframe in timeframes:
            self._resampled_data[timeframe] = self._downsample_ohlc_data(
                timeframe, self._resampled_data["1min"]
            )


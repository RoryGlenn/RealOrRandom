import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from random_ohlc import RandomOHLC 
import random
import uuid


st.set_page_config(layout="wide", page_title="Stock Prediction Game")


def initialize_session_state() -> None:
    """
    Initialize all required session state variables.

    """
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "uuid" not in st.session_state:
        st.session_state.uuid = str(uuid.uuid4())
    if "data" not in st.session_state:
        st.session_state.data = None
    if "future_price" not in st.session_state:
        st.session_state.future_price = None
    if "choices" not in st.session_state:
        st.session_state.choices = None


@st.cache_resource
def get_ohlc_generator(num_days: int, start_price: int, volatility: float, drift: float) -> RandomOHLC:
    """
    Cache the RandomOHLC generator as a resource to avoid recreating it unnecessarily.

    Parameters
    ----------
    num_days : int
        Number of days to generate data for.
    start_price : int
        Starting price of the stock.
    volatility : float
        Volatility factor.
    drift : float
        Drift factor.

    Returns
    -------
    RandomOHLC
        The cached RandomOHLC generator instance.
    """
    return RandomOHLC(
        total_days=num_days,
        start_price=start_price,
        name="StockA",
        volatility=volatility,
        drift=drift,
    )


@st.cache_data
def generate_ohlc_data(uuid_key: str, num_days: int = 90) -> pd.DataFrame:
    """
    Generate realistic OHLC data using the RandomOHLC class.

    Parameters
    ----------
    uuid_key : str
        Unique cache invalidation key.
    num_days : int, optional
        Number of days to generate data for, by default 90.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the generated OHLC data.
    """
    start_price = 100
    volatility = random.uniform(0.1, 5)
    drift = random.uniform(0.1, 5)

    ohlc_generator = get_ohlc_generator(num_days, start_price, volatility, drift)
    minutes_in_day = 1440
    ohlc_generator.create_realistic_ohlc(num_bars=minutes_in_day * num_days, frequency="1min")
    ohlc_generator.resample_timeframes()
    return ohlc_generator.resampled_data["1D"].round(2)


def prepare_new_round() -> None:
    """
    Prepare data for a new prediction round by generating OHLC data
    and calculating the predicted close price.

    """
    uuid_key = st.session_state.uuid
    df = generate_ohlc_data(uuid_key)
    future_price = df["Close"].iloc[-1]
    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.2, 0.2)), 2) for _ in range(3)]
    )
    st.session_state.data = df.iloc[:-1]
    st.session_state.future_price = future_price
    st.session_state.choices = choices


def create_candlestick_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create and configure a larger candlestick chart.

    Parameters
    ----------
    data : pd.DataFrame
        The OHLC data for the chart.

    Returns
    -------
    go.Figure
        A Plotly candlestick chart.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
    fig.update_layout(
        title="Historical Stock Prices (Last 90 Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        height=800,
        width=1400,
    )
    return fig


def display_score() -> None:
    """
    Display the user's current score.

    """
    st.subheader("Score")
    st.write(f"Correct: {st.session_state.score['right']}")
    st.write(f"Wrong: {st.session_state.score['wrong']}")


def main() -> None:
    """
    Main function for running the Streamlit app.

    """
    initialize_session_state()

    # Prepare new round only if new data is required
    if st.session_state.data is None or st.session_state.choices is None:
        prepare_new_round()

    # Streamlit UI
    st.title("Stock Price Prediction Game (90-Day Period)")

    # Disable interactivity after submission
    is_disabled = st.session_state.submitted

    # Create and display the candlestick chart
    fig = create_candlestick_chart(st.session_state.data)
    st.plotly_chart(fig, use_container_width=False)

    # Prediction game
    st.subheader("What do you think the next day's closing price will be?")
    user_choice = st.radio(
        "Choose a price:", st.session_state.choices, key="user_choice", disabled=is_disabled
    )

    # Submit button
    if st.button("Submit", disabled=is_disabled):
        st.session_state.submitted = True
        if user_choice == st.session_state.future_price:
            st.success("Correct!")
            st.session_state.score["right"] += 1
        else:
            st.error(f"Wrong! The correct answer was {st.session_state.future_price:.2f}.")
            st.session_state.score["wrong"] += 1

    # Next button to trigger the next round
    if st.session_state.submitted and st.button("Next"):
        st.session_state.submitted = False
        st.session_state.uuid = str(uuid.uuid4())  # Generate a new unique ID for cache invalidation
        prepare_new_round()

    # Display score
    display_score()


if __name__ == "__main__":
    main()

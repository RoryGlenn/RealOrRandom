import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from random_ohlc import RandomOHLC
import random
import uuid

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

# Set Streamlit layout to wide
st.set_page_config(layout="wide", page_title="Stock Prediction Game")


class GameState:
    """
    Enum class to represent the state of the game.

    Attributes:
    -----------
    INITIAL: int
        Initial state before user presses submit.
    SHOW_RESULT: int
        State after user presses submit.
    FINISHED: int
        State after n total attempts.
    """

    INITIAL = 0
    SHOW_RESULT = 1
    FINISHED = 2


def initialize_session_state() -> None:
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "game_state" not in st.session_state:
        # States:
        # 0: Before user presses submit (initial)
        # 1: After user presses submit (show result and next button)
        # 2: Results page after 5 total attempts
        st.session_state.game_state = GameState.INITIAL
    if "uuid" not in st.session_state:
        st.session_state.uuid = str(uuid.uuid4())
    if "data" not in st.session_state:
        st.session_state.data = None
    if "future_price" not in st.session_state:
        st.session_state.future_price = None
    if "choices" not in st.session_state:
        st.session_state.choices = None
    if "user_choice" not in st.session_state:
        st.session_state.user_choice = None


def get_ohlc_generator(
    num_days: int, start_price: int, volatility: float, drift: float
) -> RandomOHLC:
    return RandomOHLC(
        total_days=num_days,
        start_price=start_price,
        name="StockA",
        volatility=volatility,
        drift=drift,
    )

def generate_ohlc_data(num_days: int = 90) -> pd.DataFrame:
    start_price = 10000
    volatility = random.uniform(1, 3)
    drift = random.uniform(1, 3)

    logger.info(
        f"Num Days: {num_days}, Start Price: {start_price}, Volatility: {volatility}, Drift: {drift}"
    )

    ohlc_generator = get_ohlc_generator(num_days, start_price, volatility, drift)
    minutes_in_day = 1440
    ohlc_generator.create_realistic_ohlc(
        num_bars=minutes_in_day * num_days, frequency="1min"
    )
    ohlc_generator.resample_timeframes()
    return ohlc_generator.resampled_data["1D"].round(2)


def prepare_new_round() -> None:
    df = generate_ohlc_data()
    future_price = df["Close"].iloc[-1]
    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.1, 0.1)), 2) for _ in range(3)]
    )

    st.session_state.data = df.iloc[:-1]
    st.session_state.future_price = future_price
    st.session_state.choices = choices
    st.session_state.user_choice = None


def create_candlestick_chart(data: pd.DataFrame) -> go.Figure:
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
    st.subheader("Score")
    st.write(f"Correct: {st.session_state.score['right']}")
    st.write(f"Wrong: {st.session_state.score['wrong']}")


def submit_callback():
    user_choice = st.session_state.user_choice
    future_price = st.session_state.future_price
    if user_choice is None:
        st.warning("Please select a price before submitting.")
        return

    # Evaluate result
    if user_choice == future_price:
        st.session_state.score["right"] += 1
        st.success("Correct!")
    else:
        st.session_state.score["wrong"] += 1
        st.error(f"Wrong! The correct answer was {future_price:.2f}.")

    # Check attempts
    total_attempts = st.session_state.score["right"] + st.session_state.score["wrong"]
    if total_attempts >= 5:
        # Show results page
        st.session_state.game_state = 2
    else:
        # Move to state 1
        st.session_state.game_state = 1


def next_callback():
    st.session_state.uuid = str(uuid.uuid4())
    st.session_state.game_state = 0
    prepare_new_round()


def main():
    initialize_session_state()

    # If we've reached the results page, just show results
    if st.session_state.game_state == GameState.FINISHED:
        st.title("Final Results")
        st.write("You have completed 5 attempts.")
        display_score()
        st.write("Thank you for playing!")
        return

    # Prepare a new round if needed
    if st.session_state.data is None or (
        st.session_state.game_state == GameState.INITIAL
        and st.session_state.user_choice is None
    ):
        prepare_new_round()

    st.title("Stock Price Prediction Game (90-Day Period)")
    fig = create_candlestick_chart(st.session_state.data)
    st.plotly_chart(fig, use_container_width=False)
    display_score()

    st.subheader("What do you think the next day's closing price will be?")
    st.radio("Choose a price:", st.session_state.choices, key="user_choice")

    if st.session_state.game_state == GameState.INITIAL:
        st.button("Submit", on_click=submit_callback)
    elif st.session_state.game_state == GameState.SHOW_RESULT:
        st.info("Press Next to continue")
        st.button("Next", on_click=next_callback)


if __name__ == "__main__":
    main()

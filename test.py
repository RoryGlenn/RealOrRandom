import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from random_ohlc import RandomOHLC
import random
import uuid

# Set Streamlit layout to wide
st.set_page_config(layout="wide", page_title="Stock Prediction Game")


def initialize_session_state() -> None:
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "game_state" not in st.session_state:
        # States:
        # 0: Before user presses submit
        # 1: After user presses submit (show result and next button)
        st.session_state.game_state = 0
    if "uuid" not in st.session_state:
        st.session_state.uuid = str(uuid.uuid4())
    if "data" not in st.session_state:
        st.session_state.data = None
    if "future_price" not in st.session_state:
        st.session_state.future_price = None
    if "choices" not in st.session_state:
        st.session_state.choices = None


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
    start_price = 100
    volatility = random.uniform(0.1, 5)
    drift = random.uniform(0.1, 5)

    ohlc_generator = get_ohlc_generator(num_days, start_price, volatility, drift)
    minutes_in_day = 1440
    ohlc_generator.create_realistic_ohlc(
        num_bars=minutes_in_day * num_days, frequency="1min"
    )
    ohlc_generator.resample_timeframes()
    return ohlc_generator.resampled_data["1D"].round(2)


def prepare_new_round() -> None:
    df = generate_ohlc_data(90)
    future_price = df["Close"].iloc[-1]
    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.2, 0.2)), 2) for _ in range(3)]
    )

    st.session_state.data = df.iloc[:-1]
    st.session_state.future_price = future_price
    st.session_state.choices = choices


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


def main():
    initialize_session_state()

    # If no data or we just reset the game_state, prepare a new round
    if st.session_state.data is None or (
        st.session_state.game_state == 0 and "user_choice" not in st.session_state
    ):
        prepare_new_round()

    st.title("Stock Price Prediction Game (90-Day Period)")

    # Display the chart
    fig = create_candlestick_chart(st.session_state.data)
    st.plotly_chart(fig, use_container_width=False)

    # Display score
    display_score()

    if st.session_state.game_state == 0:
        # User hasn't submitted yet
        with st.form("prediction_form"):
            st.subheader("What do you think the next day's closing price will be?")
            user_choice = st.radio(
                "Choose a price:", st.session_state.choices, key="user_choice"
            )
            submitted = st.form_submit_button("Submit")

            if submitted:
                # Evaluate result
                if user_choice == st.session_state.future_price:
                    st.session_state.score["right"] += 1
                    st.success("Correct!")
                else:
                    st.session_state.score["wrong"] += 1
                    st.error(
                        f"Wrong! The correct answer was {st.session_state.future_price:.2f}."
                    )

                st.session_state.game_state = 1

    elif st.session_state.game_state == 1:
        # Submission done, show Next button
        st.info("Press Next to continue")
        if st.button("Next"):
            # Reset game state and regenerate data
            st.session_state.uuid = str(uuid.uuid4())
            st.session_state.game_state = 0
            if "user_choice" in st.session_state:
                del st.session_state["user_choice"]
            # No need to manually rerun; Streamlit reruns automatically after button click


if __name__ == "__main__":
    main()

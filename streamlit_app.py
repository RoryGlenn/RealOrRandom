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

st.set_page_config(layout="wide", page_title="Stock Prediction Game")


class GameState:
    START = -1
    INITIAL = 0
    SHOW_RESULT = 1
    FINISHED = 2


def initialize_session_state() -> None:
    if "score" not in st.session_state:
        st.session_state.score = {"right": 0, "wrong": 0}
    if "game_state" not in st.session_state:
        st.session_state.game_state = GameState.START
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
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "Easy"
    if "guesses" not in st.session_state:
        # Store tuples of (attempt_number, user_guess, actual_price)
        st.session_state.guesses = []


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


def generate_ohlc_data(num_days: int = 150) -> pd.DataFrame:
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
    df = ohlc_generator.resampled_data["1D"].round(2)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].clip(lower=1.0)
    return df


def prepare_new_round() -> None:
    df = generate_ohlc_data()
    display_days = 90
    last_displayed_day = display_days - 1

    if st.session_state.difficulty == "Easy":
        future_day_index = last_displayed_day + 1
    elif st.session_state.difficulty == "Medium":
        future_day_index = last_displayed_day + 7
    else:
        future_day_index = last_displayed_day + 30

    future_price = df["Close"].iloc[future_day_index]
    st.session_state.data = df.iloc[:display_days]
    st.session_state.future_price = future_price

    choices = sorted(
        [future_price]
        + [round(future_price * (1 + random.uniform(-0.1, 0.1)), 2) for _ in range(3)]
    )
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

    total_attempts = (
        st.session_state.score["right"] + st.session_state.score["wrong"] + 1
    )
    # Record the guess
    st.session_state.guesses.append((total_attempts, user_choice, future_price))

    # Evaluate result
    if user_choice == future_price:
        st.session_state.score["right"] += 1
        st.success("Correct!")
    else:
        st.session_state.score["wrong"] += 1
        st.error(f"Wrong! The correct answer was {future_price:.2f}.")

    total_attempts = st.session_state.score["right"] + st.session_state.score["wrong"]
    if total_attempts >= 5:
        st.session_state.game_state = GameState.FINISHED
    else:
        st.session_state.game_state = GameState.SHOW_RESULT


def next_callback():
    st.session_state.uuid = str(uuid.uuid4())
    st.session_state.game_state = GameState.INITIAL
    prepare_new_round()


def start_callback():
    st.session_state.game_state = GameState.INITIAL
    prepare_new_round()


def show_results_page():
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("## Final Results")
    st.write("You have completed 5 attempts.")
    display_score()

    # Compute additional metrics
    guesses_df = pd.DataFrame(
        st.session_state.guesses, columns=["Attempt", "Your Guess", "Actual Price"]
    )
    guesses_df["Absolute Error"] = (
        guesses_df["Your Guess"] - guesses_df["Actual Price"]
    ).abs()
    accuracy = (st.session_state.score["right"] / 5) * 100
    avg_error = guesses_df["Absolute Error"].mean()

    st.write(f"**Accuracy:** {accuracy:.2f}%")
    st.write(f"**Average Absolute Error:** {avg_error:.2f}")

    # Show a summary table of attempts
    st.write("### Your Attempts")
    st.dataframe(guesses_df)

    # Add a chart to visualize guesses vs. actual prices
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=guesses_df["Attempt"],
            y=guesses_df["Actual Price"],
            mode="lines+markers",
            name="Actual Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=guesses_df["Attempt"],
            y=guesses_df["Your Guess"],
            mode="lines+markers",
            name="Your Guess",
        )
    )

    fig.update_layout(
        title="Your Guesses vs Actual Prices",
        xaxis_title="Attempt Number",
        yaxis_title="Price",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Provide a final message based on performance
    if st.session_state.score["right"] > 3:
        st.write(
            "**Great job!** You got most of them right. Consider trying a harder difficulty next time!"
        )
    elif st.session_state.score["right"] == 0:
        st.write(
            "**Tough luck this time!** Consider trying again to improve your accuracy."
        )
    else:
        st.write("You did okay! With a bit more practice, you might do even better.")

    # Restart option
    st.write("### Play Again?")
    if st.button("Go Back to Start"):
        # Reset relevant states
        st.session_state.score = {"right": 0, "wrong": 0}
        st.session_state.game_state = GameState.START
        st.session_state.guesses = []

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    initialize_session_state()

    if st.session_state.game_state == GameState.START:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("## Welcome to the Stock Prediction Game!")
        st.write("You will be shown a chart of the past 90 days' prices.")
        st.write("Your goal is to guess the future closing price:")
        st.write("- **Easy:** Next day's closing price")
        st.write("- **Medium:** 7 days in the future")
        st.write("- **Hard:** 30 days in the future")
        difficulty = st.radio("Select Difficulty:", ["Easy", "Medium", "Hard"], index=0)
        st.session_state.difficulty = difficulty
        st.button("Start Game", on_click=start_callback)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.session_state.game_state == GameState.FINISHED:
        show_results_page()
        return

    if st.session_state.data is None or (
        st.session_state.game_state == GameState.INITIAL
        and st.session_state.user_choice is None
    ):
        prepare_new_round()

    st.title("Stock Price Prediction Game")
    fig = create_candlestick_chart(st.session_state.data)
    st.plotly_chart(fig, use_container_width=False)
    display_score()

    st.subheader("What do you think the future closing price will be?")
    st.radio("Choose a price:", st.session_state.choices, key="user_choice")

    if st.session_state.game_state == GameState.INITIAL:
        st.button("Submit", on_click=submit_callback)
    elif st.session_state.game_state == GameState.SHOW_RESULT:
        st.info("Press Next to continue")
        st.button("Next", on_click=next_callback)


if __name__ == "__main__":
    main()

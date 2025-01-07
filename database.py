"""Database operations for storing game results."""

# Standard library imports
import logging
import sqlite3
from typing import Any, Dict

# Configure logging
logger = logging.getLogger(__name__)


def init_db() -> None:
    """Initialize SQLite database and create tables if they don't exist."""
    try:
        with sqlite3.connect("game_data.db") as conn:
            # Enable foreign key support
            conn.execute("PRAGMA foreign_keys = ON")

            # Create game_sessions table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS game_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                correct_guesses INTEGER NOT NULL,
                wrong_guesses INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                system_info TEXT NOT NULL
            )"""
            )

            # Create guesses table
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS guesses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                attempt_number INTEGER NOT NULL,
                user_guess REAL NOT NULL,
                actual_price REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES game_sessions (id)
                    ON DELETE CASCADE
            )"""
            )

            # Verify tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not {"game_sessions", "guesses"}.issubset({t[0] for t in tables}):
                raise sqlite3.Error("Failed to create required tables")

            logger.info("Database initialized successfully")

    except sqlite3.Error as e:
        logger.error("Failed to initialize database: %s", e)
        raise


def store_game_results(results: Dict[str, Any]) -> bool:
    """
    Store game results in SQLite database.

    Parameters
    ----------
    results : Dict[str, Any]
        Game session results including scores and guesses

    Returns
    -------
    bool
        True if data was stored successfully, False otherwise
    """
    try:
        with sqlite3.connect("game_data.db") as conn:
            cursor = conn.cursor()

            # Insert session data
            cursor.execute(
                """
            INSERT INTO game_sessions 
            (timestamp, difficulty, correct_guesses, wrong_guesses, accuracy, system_info)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    results["timestamp"],
                    results["difficulty"],
                    results["score"]["right"],
                    results["score"]["wrong"],
                    results["metrics"]["accuracy"],
                    str(results["system_info"]),
                ),
            )

            session_id = cursor.lastrowid
            logger.info("Game session stored with ID: %d", session_id)

            # Insert guesses
            for attempt, guess, actual in results["guesses"]:
                cursor.execute(
                    """
                INSERT INTO guesses 
                (session_id, attempt_number, user_guess, actual_price)
                VALUES (?, ?, ?, ?)
                """,
                    (
                        session_id,
                        attempt,
                        float(guess.replace("$", "").replace(",", "")),
                        float(actual.replace("$", "").replace(",", "")),
                    ),
                )

            logger.info(
                "Stored %d guesses for session %d", len(results["guesses"]), session_id
            )
            return True

    except (sqlite3.Error, ValueError) as e:
        logger.error("Failed to store game results: %s", e)
        return False

/**
 * database.ts
 *
 * This file handles database initialization and storing game results.
 */

import sqlite3 from "sqlite3";
import { open, Database } from "sqlite";
import path from "path";
import { SessionResults } from "./types";

const DB_FILE_PATH = path.resolve(__dirname, "../game_data.db");

let db: Database<sqlite3.Database, sqlite3.Statement> | null = null;

/**
 * Initialize the SQLite database and create tables if not exist.
 */
export async function initDb(): Promise<void> {
  try {
    db = await open({
      filename: DB_FILE_PATH,
      driver: sqlite3.Database,
    });

    // Enable foreign key support
    await db.run("PRAGMA foreign_keys = ON");

    // Create game_sessions table
    await db.run(`
      CREATE TABLE IF NOT EXISTS game_sessions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp TEXT NOT NULL,
          difficulty TEXT NOT NULL,
          correct_guesses INTEGER NOT NULL,
          wrong_guesses INTEGER NOT NULL,
          accuracy REAL NOT NULL,
          system_info TEXT NOT NULL
      )
    `);

    // Create guesses table
    await db.run(`
      CREATE TABLE IF NOT EXISTS guesses (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id INTEGER NOT NULL,
          attempt_number INTEGER NOT NULL,
          user_guess REAL NOT NULL,
          actual_price REAL NOT NULL,
          FOREIGN KEY (session_id) REFERENCES game_sessions (id)
              ON DELETE CASCADE
      )
    `);

    // Optional verification of table creation
    // const tables = await db.all(`SELECT name FROM sqlite_master WHERE type='table'`);
    console.info("Database initialized successfully.");
  } catch (error) {
    console.error("Failed to initialize database:", error);
    throw error;
  }
}

/**
 * Store game results in the SQLite database.
 *
 * @param results - The SessionResults data to store.
 * @returns boolean indicating success or failure
 */
export async function storeGameResults(results: SessionResults): Promise<boolean> {
  if (!db) {
    console.error("Database not initialized. Call initDb() first.");
    return false;
  }
  const { timestamp, difficulty, score, guesses, metrics, system_info } = results;
  try {
    const insertSession = await db.run(
      `
        INSERT INTO game_sessions
        (timestamp, difficulty, correct_guesses, wrong_guesses, accuracy, system_info)
        VALUES (?, ?, ?, ?, ?, ?)
      `,
      [
        timestamp,
        difficulty,
        score.right,
        score.wrong,
        metrics.accuracy,
        JSON.stringify(system_info),
      ]
    );

    const sessionId = insertSession.lastID;
    console.info(`Game session stored with ID: ${sessionId}`);

    for (const [attempt, guess, actual] of guesses) {
      // Convert strings like "$1,234.56" to float
      const guessFloat = parseFloat(guess.replace(/\$/g, "").replace(/,/g, ""));
      const actualFloat = parseFloat(actual.replace(/\$/g, "").replace(/,/g, ""));

      await db.run(
        `
          INSERT INTO guesses
          (session_id, attempt_number, user_guess, actual_price)
          VALUES (?, ?, ?, ?)
        `,
        [sessionId, attempt, guessFloat, actualFloat]
      );
    }
    console.info(`Stored ${guesses.length} guesses for session ${sessionId}`);
    return true;
  } catch (error) {
    console.error("Failed to store game results:", error);
    return false;
  }
}

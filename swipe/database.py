"""
database.py — SQLite storage for your like/dislike ratings.

Schema (one table):
    ratings(
        set_num  TEXT PRIMARY KEY,   -- e.g. "75192-1"
        liked    INTEGER NOT NULL,   -- 1 = like, 0 = dislike
        rated_at TEXT NOT NULL       -- ISO timestamp, useful for "undo" and evaluation
    )

set_num is the PRIMARY KEY, so each set can only have one rating. Rating a set
again overwrites the old rating (an "upsert") instead of creating a duplicate.

Why SQLite? It's a full SQL database stored in a single file (db/ratings.db),
built into Python's standard library — no server to install or run.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Overridable via env var so tests can use a throwaway database file.
DEFAULT_DB_PATH = Path("db") / "ratings.db"


def get_db_path() -> Path:
    return Path(os.environ.get("LEGO_RATINGS_DB_PATH", DEFAULT_DB_PATH))


@contextmanager
def _connect():
    """
    Open a connection, commit on success, roll back on error, always close.

    Using `with _connect() as conn:` guarantees we never leave a half-written
    transaction or a dangling open file handle.
    """
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the ratings table if it doesn't exist. Safe to call every startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                set_num  TEXT PRIMARY KEY,
                liked    INTEGER NOT NULL CHECK (liked IN (0, 1)),
                rated_at TEXT NOT NULL
            )
            """
        )


def save_rating(set_num: str, liked: bool) -> None:
    """Insert or overwrite the rating for one set."""
    if not isinstance(set_num, str) or not set_num.strip():
        raise ValueError("set_num must be a non-empty string")

    # Microsecond precision so two quick ratings never get the same timestamp
    # (which would make "undo last" ambiguous).
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    with _connect() as conn:
        # "?" placeholders let sqlite3 escape values safely — never build SQL
        # with f-strings from user data (that's how SQL injection happens).
        conn.execute(
            """
            INSERT INTO ratings (set_num, liked, rated_at) VALUES (?, ?, ?)
            ON CONFLICT(set_num) DO UPDATE SET liked = excluded.liked,
                                               rated_at = excluded.rated_at
            """,
            (set_num, int(bool(liked)), now),
        )


def delete_rating(set_num: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM ratings WHERE set_num = ?", (set_num,))


def get_ratings() -> pd.DataFrame:
    """All ratings, oldest first. Columns: set_num, liked (0/1), rated_at."""
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT set_num, liked, rated_at FROM ratings ORDER BY rated_at",
            conn,
        )


def get_last_rating() -> Optional[dict]:
    """The most recent rating as a dict, or None if nothing has been rated."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT set_num, liked, rated_at FROM ratings ORDER BY rated_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return {"set_num": row[0], "liked": row[1], "rated_at": row[2]}


def rating_counts() -> dict:
    """{'likes': int, 'dislikes': int, 'total': int}"""
    with _connect() as conn:
        likes, dislikes = conn.execute(
            "SELECT COALESCE(SUM(liked), 0), COALESCE(SUM(1 - liked), 0) FROM ratings"
        ).fetchone()
    return {"likes": likes, "dislikes": dislikes, "total": likes + dislikes}


def clear_ratings() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM ratings")

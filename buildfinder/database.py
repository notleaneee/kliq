"""
database.py — SQLite connection and schema.

Everything lives in one file, data/lego.db, split into two groups of tables:

  Catalog tables (rebuilt from Rebrickable data by build_finder_catalog.py):
    colors        id -> name, rgb
    part_categories
    parts         part_num -> name, category
    part_colors   which colours each part actually exists in (+ image URL)
    builds        one row per candidate build (name, image, instructions link, ...)
    build_parts   the parts list of each build: (build_id, part_num, color_id, quantity)

    sets          every official set (for "import a set I own")
    set_parts     every set's full parts list, spares marked separately
    part_relationships  Rebrickable's links between parts (molds, prints, ...),
                  used for substitutions

  User tables (your data — never touched when the catalog is rebuilt):
    inventory     (part_num, color_id) -> quantity you OWN, notes
    projects      builds you've reserved pieces for, built, or taken apart
    project_parts what each project needs, and how much of it is covered
    project_holds the physical pieces each project is holding

  Owned vs available: pieces held by an active project (reserved or built)
  are still yours, but they're not free for another build:
      available = owned - held

A LEGO piece is identified by the PAIR (part_num, color_id): a red 2x4 brick
(3001, Red) and a blue 2x4 brick (3001, Blue) are different inventory items.
That pair is the PRIMARY KEY of the inventory table, which makes it impossible
to store the same piece twice. Adding more of a piece you own updates the
existing row instead.
"""

import gzip
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "lego.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS colors (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    rgb       TEXT,
    is_trans  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS part_categories (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parts (
    part_num     TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    part_cat_id  INTEGER,
    img_url      TEXT,                       -- picture, for the piece picker
    num_sets     INTEGER NOT NULL DEFAULT 0  -- how many sets use it (common first)
);

CREATE TABLE IF NOT EXISTS part_colors (
    part_num  TEXT NOT NULL,
    color_id  INTEGER NOT NULL,
    img_url   TEXT,
    num_sets  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (part_num, color_id)
);

CREATE TABLE IF NOT EXISTS builds (
    build_id            TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    year                INTEGER,
    theme               TEXT,
    num_parts           INTEGER NOT NULL,
    img_url             TEXT,
    instructions_url    TEXT,
    instructions_note   TEXT,
    difficulty          TEXT,
    build_time_minutes  INTEGER,
    source              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS build_parts (
    build_id  TEXT NOT NULL,
    part_num  TEXT NOT NULL,
    color_id  INTEGER NOT NULL,
    quantity  INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (build_id, part_num, color_id)
);
CREATE INDEX IF NOT EXISTS idx_build_parts_piece ON build_parts (part_num, color_id);

CREATE TABLE IF NOT EXISTS sets (
    set_num    TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    year       INTEGER,
    theme      TEXT,
    num_parts  INTEGER,
    img_url    TEXT
);

CREATE TABLE IF NOT EXISTS set_parts (
    set_num   TEXT NOT NULL,
    part_num  TEXT NOT NULL,
    color_id  INTEGER NOT NULL,
    is_spare  INTEGER NOT NULL CHECK (is_spare IN (0, 1)),
    quantity  INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (set_num, part_num, color_id, is_spare)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS inventory (
    part_num    TEXT NOT NULL,
    color_id    INTEGER NOT NULL,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    notes       TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (part_num, color_id)
);

CREATE TABLE IF NOT EXISTS projects (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id       TEXT NOT NULL,
    name           TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('reserved', 'built', 'disassembled', 'cancelled')),
    pieces_needed  INTEGER NOT NULL,
    notes          TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    released_at    TEXT            -- NULL while the project is holding pieces
);

-- What the build needs, line by line, and how many of each line are covered.
CREATE TABLE IF NOT EXISTS project_parts (
    project_id  INTEGER NOT NULL REFERENCES projects (id),
    part_num    TEXT NOT NULL,
    color_id    INTEGER NOT NULL,
    needed      INTEGER NOT NULL CHECK (needed > 0),
    held        INTEGER NOT NULL CHECK (held >= 0 AND held <= needed),
    PRIMARY KEY (project_id, part_num, color_id)
);

-- The actual physical pieces taken from your inventory. Usually the same as
-- the needed piece, but can be a substitute (another mold, an old grey...).
CREATE TABLE IF NOT EXISTS project_holds (
    project_id  INTEGER NOT NULL REFERENCES projects (id),
    part_num    TEXT NOT NULL,
    color_id    INTEGER NOT NULL,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (project_id, part_num, color_id)
);

CREATE TABLE IF NOT EXISTS part_relationships (
    rel_type         TEXT NOT NULL,
    child_part_num   TEXT NOT NULL,
    parent_part_num  TEXT NOT NULL
);
"""

# Migration for databases created in Stage 2, before project_holds existed:
# back then every held piece was exactly the needed piece.
_MIGRATE_HOLDS_SQL = """
INSERT INTO project_holds (project_id, part_num, color_id, quantity)
SELECT pp.project_id, pp.part_num, pp.color_id, pp.held
FROM project_parts pp
WHERE pp.held > 0
  AND NOT EXISTS (SELECT 1 FROM project_holds ph WHERE ph.project_id = pp.project_id)
"""

# Pieces currently held by projects. A project stops holding pieces when it's
# released (cancelled, disassembled, or completed without tracking).
HELD_PIECES_SQL = """
    SELECT ph.part_num, ph.color_id, SUM(ph.quantity) AS held
    FROM project_holds ph JOIN projects p ON p.id = ph.project_id
    WHERE p.released_at IS NULL
    GROUP BY ph.part_num, ph.color_id
"""

CATALOG_TABLES = ["colors", "part_categories", "parts", "part_colors", "builds", "build_parts",
                  "sets", "set_parts", "part_relationships"]


def get_db_path() -> Path:
    # Overridable so tests can use a throwaway database.
    return Path(os.environ.get("LEGO_INVENTORY_DB_PATH", DEFAULT_DB_PATH))


def unpack_if_needed() -> None:
    """
    The hosted copy ships data/lego.db.gz (GitHub refuses files over 100 MB).
    Unpack it once, the first time the app starts. Written to a temp file and
    renamed, so two visitors arriving together never see a half-written file.
    """
    path = get_db_path()
    packed = path.with_name(path.name + ".gz")
    if path.exists() or not packed.exists():
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(packed, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp, path)


@contextmanager
def connect():
    """Open a connection; commit on success, roll back on error, always close."""
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


# Columns added after a table was first created. CREATE TABLE IF NOT EXISTS
# doesn't touch existing tables, so older databases get them via ALTER TABLE.
_ADDED_COLUMNS = {
    "parts": {"img_url": "TEXT", "num_sets": "INTEGER NOT NULL DEFAULT 0"},
    "part_colors": {"num_sets": "INTEGER NOT NULL DEFAULT 0"},
}


def init_db() -> None:
    """Create any missing tables and columns. Safe to call on every startup."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, definition in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.execute(_MIGRATE_HOLDS_SQL)


def catalog_is_loaded() -> bool:
    with connect() as conn:
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM builds").fetchone()
        except sqlite3.OperationalError:
            return False
    return n > 0

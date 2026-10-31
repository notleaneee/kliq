"""
data_loader.py — Read the raw Rebrickable CSV downloads.

Source: https://rebrickable.com/downloads/ (free, no API key needed).
Files used and what they contain:

  colors.csv            colour id -> name (e.g. 4 -> "Red")
  parts.csv             part_num -> name (e.g. "3001" -> "Brick 2 x 4")
  part_categories.csv   category id -> name (e.g. 11 -> "Bricks")
  sets.csv              official sets: set_num, name, year, theme_id, num_parts, img_url
  themes.csv            theme id -> name
  inventories.csv       an "inventory" is one parts list. id, version, set_num.
                        Minifigs have inventories too (set_num like "fig-000123").
  inventory_parts.csv   the rows of each parts list: inventory_id, part_num,
                        color_id, quantity, is_spare, img_url
  inventory_minifigs.csv  minifigs included in an inventory
  inventory_sets.csv    sets included inside other sets (multi-packs)
  part_relationships.csv  links between parts: M = mold variant, A = alternate,
                        P = print, T = pattern, R = pair, B = sub-part

Lookup order for the folder: $LEGO_RAW_DIR, then all_downloads/, then data/raw.
"""

import os
from pathlib import Path
from typing import Dict

import pandas as pd

from buildfinder.database import PROJECT_ROOT

REQUIRED_FILES = [
    "colors.csv", "parts.csv", "part_categories.csv", "sets.csv", "themes.csv",
    "inventories.csv", "inventory_parts.csv", "inventory_minifigs.csv", "inventory_sets.csv",
    "part_relationships.csv",
]

# Read IDs that look numeric (like part "3001") as text so "3001" and
# "03001" never get merged and leading zeros are kept.
_STR_COLUMNS = {"part_num": str, "set_num": str, "fig_num": str,
                "child_part_num": str, "parent_part_num": str}


def get_raw_dir() -> Path:
    candidates = []
    if os.environ.get("LEGO_RAW_DIR"):
        candidates.append(Path(os.environ["LEGO_RAW_DIR"]))
    candidates += [PROJECT_ROOT / "all_downloads", PROJECT_ROOT / "data" / "raw"]

    for directory in candidates:
        if (directory / "sets.csv").exists():
            return directory
    raise FileNotFoundError(
        "Rebrickable CSVs not found. Download them from https://rebrickable.com/downloads/ "
        f"and unzip into one of: {', '.join(str(c) for c in candidates)}"
    )


def load_raw(raw_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """Load every required CSV into a dict keyed by file stem, e.g. raw['sets']."""
    raw_dir = raw_dir or get_raw_dir()
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {raw_dir}: {', '.join(missing)}")

    raw = {}
    for filename in REQUIRED_FILES:
        df = pd.read_csv(raw_dir / filename, dtype=_STR_COLUMNS, low_memory=False)
        raw[filename[:-4]] = df
        print(f"  {filename:<24} {len(df):>10,} rows")
    return raw

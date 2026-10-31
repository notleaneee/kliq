"""
data_loader.py — Loads LEGO set metadata from Rebrickable CSVs and downloads images.

Key concepts:
  - sets.csv has one row per LEGO set with: set_num, name, year, theme_id, num_parts, img_url
  - themes.csv maps theme_id -> theme name (e.g. "Technic", "City")
  - We merge them so every set row carries its human-readable theme name
  - Images are downloaded once and cached to data/images/ by set_num
"""

import io
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm

# Root paths — relative to wherever the script is run from (always the project root)
DATA_DIR = Path("data")
ALL_DOWNLOADS_DIR = Path("all_downloads")   # the CSVs you already have
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_sets(min_parts: int = 10, max_rows: int = None) -> pd.DataFrame:
    """
    Load and clean the sets + themes CSVs, returning a merged DataFrame.

    Parameters
    ----------
    min_parts : int
        Drop sets with fewer pieces — tiny polybags and keychains add noise
        to the model without giving meaningful visual variety.
    max_rows : int or None
        Cap the number of rows. Useful during development so you're not
        waiting on 20k image downloads. Set to None for the full dataset.

    Returns
    -------
    pd.DataFrame with columns:
        set_num, name, year, theme_id, num_parts, img_url, theme_name
    """
    # Try the pre-downloaded folder first, then the standard data/ folder
    sets_path = _find_csv("sets.csv")
    themes_path = _find_csv("themes.csv")

    sets_df = pd.read_csv(sets_path)
    themes_df = (
        pd.read_csv(themes_path)[["id", "name"]]
        .rename(columns={"id": "theme_id", "name": "theme_name"})
    )

    # Left-join so every set row gets its theme name (NaN if theme missing)
    df = sets_df.merge(themes_df, on="theme_id", how="left")

    # --- Cleaning ---
    df = df.dropna(subset=["img_url"])                          # no image = useless for CLIP
    df["num_parts"] = pd.to_numeric(df["num_parts"], errors="coerce").fillna(0).astype(int)
    df = df[df["num_parts"] >= min_parts]                       # filter tiny sets
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["theme_name"] = df["theme_name"].fillna("Unknown")
    df = df.reset_index(drop=True)

    if max_rows:
        df = df.head(max_rows)

    print(f"Loaded {len(df):,} sets across {df['theme_name'].nunique()} themes "
          f"({df['year'].min()}–{df['year'].max()})")
    return df


def download_image(img_url: str, set_num: str, timeout: int = 10) -> Optional[Path]:
    """
    Download a single image to data/images/<set_num>.<ext>.

    Returns the local Path on success, None on any failure.
    Skips the network call entirely if the file already exists (cache hit).
    """
    # Derive file extension from the URL (default to jpg if URL has no dot)
    ext = img_url.rsplit(".", 1)[-1].lower() if "." in img_url.split("/")[-1] else "jpg"
    local_path = IMAGES_DIR / f"{set_num}.{ext}"

    if local_path.exists():
        return local_path   # already cached

    try:
        response = requests.get(img_url, timeout=timeout)
        response.raise_for_status()

        # PIL.verify() confirms the bytes are a valid image before we save them.
        # Without this check a truncated download would silently corrupt the cache.
        img = Image.open(io.BytesIO(response.content))
        img.verify()

        local_path.write_bytes(response.content)
        return local_path

    except Exception as e:
        print(f"  [warn] Failed to download {set_num}: {e}")
        return None


def download_images_batch(
    df: pd.DataFrame,
    limit: int = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    """
    Download images for every row in df (or just the first `limit` rows).

    Parameters
    ----------
    df          : DataFrame from load_sets()
    limit       : only process the first N rows — handy for quick tests
    max_workers : parallel downloads. Each download spends most of its time
                  waiting on the network, so threads give a big speed-up.
                  Kept small to be polite to Rebrickable's CDN.

    Returns a copy of df with a new column 'local_image_path' (str or None).
    Sets whose image failed to download get None — they'll be skipped later.
    """
    subset = df.head(limit) if limit else df
    jobs = list(zip(subset.index, subset["img_url"], subset["set_num"]))
    paths = {}

    print(f"Downloading images for {len(subset):,} sets "
          f"(skipping already-cached)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_image, url, set_num): idx
            for idx, url, set_num in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="img"):
            path = future.result()
            paths[futures[future]] = str(path) if path else None

    df = df.copy()
    df["local_image_path"] = None
    df.loc[subset.index, "local_image_path"] = [paths[i] for i in subset.index]
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_csv(filename: str) -> Path:
    """
    Look for a CSV in all_downloads/ first (your pre-downloaded files),
    then fall back to data/. Raises FileNotFoundError with a helpful message.
    """
    for directory in [ALL_DOWNLOADS_DIR, DATA_DIR]:
        path = directory / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find '{filename}' in all_downloads/ or data/.\n"
        "Download it from: https://rebrickable.com/downloads/"
    )

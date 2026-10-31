"""
catalog.py — The pool of LEGO sets the app shows you.

The "catalog" is the subset of sets that have BOTH a downloaded image AND a
CLIP embedding. Restricting the app to these sets guarantees that every set
you rate can be used to train the model in Stage 4.

Building it ties Stages 1 and 2 together:
    load_sets()  ->  random sample  ->  download images  ->  CLIP embeddings
and writes the result to data/catalog.csv so the app can load it instantly.
"""

import os
from pathlib import Path

import pandas as pd

from swipe.data_loader import load_sets, download_images_batch

DEFAULT_CATALOG_PATH = Path("data") / "catalog.csv"


def get_catalog_path() -> Path:
    # Overridable so tests can point the app at a small fake catalog.
    return Path(os.environ.get("LEGO_CATALOG_PATH", DEFAULT_CATALOG_PATH))

# Themes that are mostly not "real" sets (loose parts, books, merchandise).
EXCLUDED_THEMES = {
    "Bulk Bricks",
    "Service Packs",
    "Activity Books with LEGO Parts",
    "Ideas Books",
    "Story Books",
    "Gear",
    "Key Chain",
}

CATALOG_COLUMNS = ["set_num", "name", "year", "theme_name", "num_parts", "img_url", "local_image_path"]


def build_catalog(n_sets: int = 500, min_parts: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Sample n_sets sets, download their images, embed them, and save the catalog.

    Re-running with a larger n_sets grows the catalog: the random sample with
    the same seed keeps the earlier sets, and cached images/embeddings are reused.
    """
    # Imported here so the app can load the catalog without pulling in PyTorch.
    from swipe.embeddings import generate_embeddings

    df = load_sets(min_parts=min_parts)
    df = df[~df["theme_name"].isin(EXCLUDED_THEMES)]

    n_sets = min(n_sets, len(df))
    # Shuffle the whole pool once with a fixed seed, then take the first n.
    # A fixed seed means the same sets come out every time, and taking a
    # prefix means growing n_sets only ever ADDS sets.
    sample = df.sample(frac=1.0, random_state=seed).head(n_sets).reset_index(drop=True)
    print(f"Sampled {len(sample):,} sets for the catalog.")

    sample = download_images_batch(sample)
    _, index_df = generate_embeddings(sample)

    catalog = sample[sample["set_num"].isin(index_df["set_num"])][CATALOG_COLUMNS]
    catalog = catalog.reset_index(drop=True)

    path = get_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    print(f"Catalog saved: {len(catalog):,} sets → {path}")
    return catalog


def load_catalog() -> pd.DataFrame:
    """Load the saved catalog. Raises FileNotFoundError with next steps if missing."""
    path = get_catalog_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first with:  python build_swipe_catalog.py"
        )
    try:
        df = pd.read_csv(path, dtype={"set_num": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ValueError(f"{path} can't be read ({e}). Rebuild with:  python build_swipe_catalog.py") from e
    missing = [c for c in CATALOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Rebuild with:  python build_swipe_catalog.py")
    # Drop rows whose image file has since been deleted from disk.
    df = df[df["local_image_path"].map(lambda p: isinstance(p, str) and Path(p).exists())]
    return df.reset_index(drop=True)

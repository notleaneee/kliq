"""
recommendations.py — Rank builds by how well they fit your inventory.

Stage 1 ranking is purely compatibility-based (no personal preferences yet).
Builds are sorted by these keys, in order; each one only breaks ties left by
the one before:

  1. status        fully buildable  >  almost buildable  >  needs more pieces
  2. coverage      higher share of required pieces you own comes first
  3. missing       fewer missing pieces first
  4. pieces used   builds that use MORE of your pieces first (less left unused)
  5. build_id      only so the order is stable between runs

So a fully buildable build always ranks above one that needs extra pieces,
and among fully buildable builds, bigger ones (using more of your
inventory) come first.
"""

from typing import Iterable, Optional

import pandas as pd

from buildfinder.compatibility import STATUS_BUILDABLE, STATUS_ORDER


def rank_builds(
    scores: pd.DataFrame,
    builds: pd.DataFrame,
    statuses: Optional[Iterable[str]] = None,
    max_missing: Optional[int] = None,
    min_parts: Optional[int] = None,
    max_parts: Optional[int] = None,
    search: str = "",
    themes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Join scores with build metadata, filter, and sort.

    scores : output of compatibility.score_builds()
    builds : the builds table (build_id, name, theme, num_parts, ...)
    """
    df = builds.merge(scores, on="build_id", how="inner")

    if statuses is not None:
        df = df[df["status"].isin(list(statuses))]
    if max_missing is not None:
        df = df[df["missing_total"] <= max_missing]
    if min_parts is not None:
        df = df[df["required_total"] >= min_parts]
    if max_parts is not None:
        df = df[df["required_total"] <= max_parts]
    if themes:
        df = df[df["theme"].isin(list(themes))]
    if search and search.strip():
        needle = search.strip().lower()
        df = df[df["name"].str.lower().str.contains(needle, regex=False)
                | df["build_id"].str.lower().str.contains(needle, regex=False)]

    df = df.assign(_status_rank=df["status"].map(STATUS_ORDER))
    df = df.sort_values(
        ["_status_rank", "coverage", "missing_total", "available_total", "build_id"],
        ascending=[True, False, True, False, True],
    )
    return df.drop(columns="_status_rank").reset_index(drop=True)


def buildable_share_at_k(ranked: pd.DataFrame, k: int = 10) -> Optional[float]:
    """
    Evaluation metric: what fraction of the top-k recommendations can you
    actually build right now? Returns None if there are no recommendations.
    """
    top = ranked.head(k)
    if top.empty:
        return None
    return float((top["status"] == STATUS_BUILDABLE).mean())


def status_counts(scores: pd.DataFrame) -> dict:
    counts = scores["status"].value_counts().to_dict()
    return {status: int(counts.get(status, 0)) for status in STATUS_ORDER}


# ── Swipe likes ──────────────────────────────────────────────────────────────
# The Swipe page stores a like/dislike per set number, and an official build's
# build_id IS its set number ("75192-1"), so the two line up directly.

SWIPE_IGNORE, SWIPE_LIKED_FIRST, SWIPE_LIKED_ONLY, SWIPE_HIDE_DISLIKED = (
    "ignore", "liked_first", "liked_only", "hide_disliked")
SWIPE_MODES = (SWIPE_IGNORE, SWIPE_LIKED_FIRST, SWIPE_LIKED_ONLY, SWIPE_HIDE_DISLIKED)


def apply_swipe_taste(ranked: pd.DataFrame, liked: Iterable[str], disliked: Iterable[str],
                      mode: str = SWIPE_IGNORE) -> pd.DataFrame:
    """
    Re-rank or filter already-ranked builds using your Swipe ratings.

    liked_first    liked builds move to the top; otherwise the order is unchanged
                   (a stable sort, so ties keep their compatibility order)
    liked_only     only builds you liked
    hide_disliked  drop builds you disliked
    """
    if mode not in SWIPE_MODES:
        raise ValueError(f"Unknown mode {mode!r}")
    liked, disliked = set(liked), set(disliked)
    if mode == SWIPE_LIKED_ONLY:
        ranked = ranked[ranked["build_id"].isin(liked)]
    elif mode == SWIPE_HIDE_DISLIKED:
        ranked = ranked[~ranked["build_id"].isin(disliked)]
    elif mode == SWIPE_LIKED_FIRST:
        is_liked = ranked["build_id"].isin(liked)
        ranked = ranked.iloc[(~is_liked).argsort(kind="stable")]
    return ranked.reset_index(drop=True)

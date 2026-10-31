"""
piece_usage.py — Make the most of the pieces you have (waste reduction).

Four questions, each answered from your AVAILABLE pieces:

  1. plan_builds        Which SEVERAL builds can I make at the same time?
                        Every physical piece is used at most once across the plan.
  2. inventory_share    Which builds use the biggest share of my pieces?
  3. unlockable_pieces  Which few extra pieces would complete the most builds?
  4. rarely_used_pieces Which of my pieces hardly any build (or recommendation) uses?

About plan_builds: choosing the best combination of builds is a hard
optimisation problem (like packing a knapsack), so this uses a simple greedy
method: go through candidate builds in order, keep each one that still fits
in what's left. It's fast and usually good, but not guaranteed to be the
best possible plan.
"""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from buildfinder.compatibility import (EXACT, GROUP_KEY, STATUS_ALMOST, STATUS_BUILDABLE, MatchRules,
                               add_group_keys, allocate, normalize_pieces)

PLAN_MOST_PIECES = "most_pieces"
PLAN_MOST_BUILDS = "most_builds"


@dataclass
class Plan:
    builds: pd.DataFrame        # build_id, pieces (in the order they were chosen)
    pieces_used: int
    pieces_available: int

    @property
    def share_used(self) -> float:
        return self.pieces_used / self.pieces_available if self.pieces_available else 0.0


def plan_builds(candidates: pd.DataFrame, build_parts: pd.DataFrame, inventory: pd.DataFrame,
                rules: MatchRules = EXACT, objective: str = PLAN_MOST_PIECES,
                max_candidates: int = 300, max_builds: int = 50) -> Plan:
    """
    Greedy plan of builds you can make at once from `inventory`.

    candidates : scores rows (build_id, required_total, status), usually from score_builds
    objective  : most_pieces -> try big builds first (uses more of your pieces)
                 most_builds -> try small builds first (fits more builds)
    """
    if objective not in (PLAN_MOST_PIECES, PLAN_MOST_BUILDS):
        raise ValueError(f"unknown objective '{objective}'")
    pool = candidates[candidates["status"] == STATUS_BUILDABLE]
    pool = pool.sort_values(["required_total", "build_id"],
                            ascending=[objective == PLAN_MOST_BUILDS, True]).head(max_candidates)

    remaining = normalize_pieces(inventory)
    total = int(remaining["quantity"].sum())
    free = {(p, c): q for p, c, q in remaining.itertuples(index=False, name=None)}
    parts_by_build = {bid: rows for bid, rows in
                      build_parts[build_parts["build_id"].isin(pool["build_id"])].groupby("build_id")}

    chosen = []
    for build_id in pool["build_id"]:
        if len(chosen) >= max_builds:
            break
        left = pd.DataFrame([(p, c, q) for (p, c), q in free.items() if q > 0],
                            columns=["part_num", "color_id", "quantity"])
        result = allocate(parts_by_build[build_id], left, rules)
        if result.missing_total == 0:
            for p, c, q in result.holds.itertuples(index=False, name=None):
                free[(p, c)] -= q
            chosen.append((build_id, result.required_total))

    plan = pd.DataFrame(chosen, columns=["build_id", "pieces"])
    return Plan(plan, int(plan["pieces"].sum()), total)


def inventory_share(ranked: pd.DataFrame, pieces_available: int, limit: int = 10) -> pd.DataFrame:
    """Buildable / almost-buildable builds, by how many of your available pieces they'd use."""
    if pieces_available <= 0:
        return ranked.head(0).assign(share=[])
    near = ranked[ranked["status"].isin([STATUS_BUILDABLE, STATUS_ALMOST])]
    near = near.sort_values(["available_total", "build_id"], ascending=[False, True]).head(limit)
    return near.assign(share=near["available_total"] / pieces_available)


def unlockable_pieces(lines: pd.DataFrame, scores: pd.DataFrame, max_missing: int = 10) -> pd.DataFrame:
    """
    Pieces that would complete builds you're close to.

    lines  : output of compatibility.group_lines()
    scores : output of compatibility.score_builds()
    Looks at builds that aren't buildable yet but miss at most `max_missing` pieces.

    Returns per piece (part_key, color_key):
      builds_unlocked  builds for which this is the ONLY missing piece
      pieces_needed    how many you'd need to complete all of those builds
                       (each build checked on its own, so the biggest single need)
      builds_helped    builds where this piece is one of the missing ones
    """
    near = scores[(scores["status"] != STATUS_BUILDABLE) & (scores["missing_total"] <= max_missing)]
    short = lines[lines["build_id"].isin(near["build_id"]) & (lines["missing"] > 0)]
    columns = GROUP_KEY + ["builds_unlocked", "pieces_needed", "builds_helped"]
    if short.empty:
        return pd.DataFrame(columns=columns)

    shortages_per_build = short.groupby("build_id")["missing"].size()
    single = shortages_per_build[shortages_per_build == 1].index
    unlock = (short[short["build_id"].isin(single)]
              .groupby(GROUP_KEY).agg(builds_unlocked=("build_id", "nunique"),
                                      pieces_needed=("missing", "max")))
    helped = short.groupby(GROUP_KEY).agg(builds_helped=("build_id", "nunique"))
    out = helped.join(unlock, how="left").reset_index()
    out["builds_unlocked"] = out["builds_unlocked"].fillna(0).astype(int)
    out["pieces_needed"] = out["pieces_needed"].fillna(0).astype(int)
    out = out.sort_values(["builds_unlocked", "builds_helped", "pieces_needed"],
                          ascending=[False, False, True])
    return out[columns].reset_index(drop=True)


def rarely_used_pieces(inventory: pd.DataFrame, build_parts: pd.DataFrame,
                       top_build_ids: Iterable[str], rules: MatchRules = EXACT) -> pd.DataFrame:
    """
    Your pieces, least useful first.

    Returns part_num, color_id, quantity, builds_using (whole catalog),
    top_builds_using (among your top recommendations).
    """
    inv = add_group_keys(normalize_pieces(inventory), rules)
    if inv.empty:
        return inv.assign(builds_using=[], top_builds_using=[])
    bp = add_group_keys(build_parts, rules)
    catalog = bp.groupby(GROUP_KEY)["build_id"].nunique().rename("builds_using")
    top = (bp[bp["build_id"].isin(list(top_build_ids))]
           .groupby(GROUP_KEY)["build_id"].nunique().rename("top_builds_using"))
    out = inv.join(catalog, on=GROUP_KEY).join(top, on=GROUP_KEY)
    out[["builds_using", "top_builds_using"]] = out[["builds_using", "top_builds_using"]].fillna(0).astype(int)
    out = out.sort_values(["top_builds_using", "builds_using", "quantity"],
                          ascending=[True, True, False])
    return out[["part_num", "color_id", "quantity", "builds_using", "top_builds_using"]].reset_index(drop=True)

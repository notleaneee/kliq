"""
compatibility.py — Compare what a build needs with what you own.

The core idea, for ONE line of a parts list:

    required  = how many of (part, colour) the build needs
    available = how many of those you can cover with your pieces
    missing   = required - available

Summing over every line gives the build's totals:

    coverage  = available_total / required_total    (0.0 - 1.0)

Matching rules (see substitutions.py for where they come from):
  * EXACT (default): a piece matches only if part_num AND color_id are equal.
  * Substitutions: parts can be grouped (e.g. two molds of the same brick)
    and colours can be grouped (e.g. old grey ~ new grey, or "any colour").
    Some rules are one-way: a plain part may cover a printed one, not the
    reverse. Each owned piece belongs to one group; each needed piece points
    to the group that may cover it.

How pieces are assigned to a build (allocate):
  1. Exact pieces first, line by line.
  2. Then, for anything still missing, pieces from the group that may cover
     it, preferring the same part in another colour, then the same colour in
     another part.

Statuses:
  buildable    nothing missing
  almost       at most ALMOST_MAX_MISSING pieces missing AND coverage >= ALMOST_MIN_COVERAGE
  needs_parts  everything else (including builds you own none of the pieces for)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

STATUS_BUILDABLE = "buildable"
STATUS_ALMOST = "almost"
STATUS_NEEDS_PARTS = "needs_parts"
STATUS_ORDER = {STATUS_BUILDABLE: 0, STATUS_ALMOST: 1, STATUS_NEEDS_PARTS: 2}
STATUS_LABELS = {
    STATUS_BUILDABLE: "Fully buildable",
    STATUS_ALMOST: "Almost buildable (needs a few pieces)",
    STATUS_NEEDS_PARTS: "Needs more pieces",
}

ALMOST_MAX_MISSING = 5
ALMOST_MIN_COVERAGE = 0.90

KEY = ["part_num", "color_id"]
GROUP_KEY = ["part_key", "color_key"]

# Colour key used when colours are ignored. Rebrickable uses -1 for
# "[Unknown]", so -2 can't clash with a real colour.
ANY_COLOR = -2


@dataclass(frozen=True)
class MatchRules:
    """
    Which pieces count as the same. Unlisted parts/colours only match themselves.

    part_map       two-way groups: any member can replace any other (mold variants)
    need_part_map  ONE-way: a part the build needs -> the part that may replace it.
                   Used for prints: a needed printed part can be covered by its
                   plain version, but a printed piece you own can't replace
                   another print (a different face, a different logo).
    color_map      two-way colour groups; any_color ignores colour entirely
    """
    part_map: Mapping[str, str] = field(default_factory=dict)
    color_map: Mapping[int, int] = field(default_factory=dict)
    any_color: bool = False
    need_part_map: Mapping[str, str] = field(default_factory=dict)

    def part_key(self, part_num: str) -> str:
        """Group of a piece you OWN."""
        return self.part_map.get(part_num, part_num)

    def need_part_key(self, part_num: str) -> str:
        """Group of pieces that can cover a piece a build NEEDS."""
        return self.part_key(self.need_part_map.get(part_num, part_num))

    def color_key(self, color_id: int) -> int:
        return ANY_COLOR if self.any_color else self.color_map.get(color_id, color_id)


EXACT = MatchRules()
ANY_COLOUR_RULES = MatchRules(any_color=True)


def normalize_pieces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a pieces table: consistent types, duplicates combined, zero rows dropped.

    "Duplicates combined" matters: if a list says (3001, Red) x 4 on one row
    and (3001, Red) x 6 on another, that's 10 red 2x4 bricks.
    """
    extra = [c for c in ("build_id",) if c in df.columns]
    if df.empty:
        # Typed empty columns, so later merges on color_id (int) still work.
        empty = {c: pd.Series(dtype=str) for c in extra + ["part_num"]}
        empty.update(color_id=pd.Series(dtype=int), quantity=pd.Series(dtype=int))
        return pd.DataFrame(empty)
    out = df[extra + KEY + ["quantity"]].copy()
    out["part_num"] = out["part_num"].astype(str)
    out["color_id"] = out["color_id"].astype(int)
    out["quantity"] = out["quantity"].astype(int)
    out = out.groupby(extra + KEY, as_index=False)["quantity"].sum()
    return out[out["quantity"] > 0].reset_index(drop=True)


def _map(series: pd.Series, mapping: Mapping) -> pd.Series:
    return series.map(mapping).fillna(series) if mapping else series


def add_group_keys(df: pd.DataFrame, rules: MatchRules, needed: bool = False) -> pd.DataFrame:
    """
    Add part_key / color_key columns: which group each piece belongs to.
    needed=True gives the key for a piece a build NEEDS (one-way rules apply).
    """
    out = df.copy()
    parts = _map(out["part_num"], rules.need_part_map) if needed else out["part_num"]
    out["part_key"] = _map(parts, rules.part_map)
    if rules.any_color:
        out["color_key"] = ANY_COLOR
    elif rules.color_map:
        out["color_key"] = out["color_id"].map(rules.color_map).fillna(out["color_id"]).astype(int)
    else:
        out["color_key"] = out["color_id"]
    return out


def classify(required_total: int, missing_total: int,
             max_missing: int = ALMOST_MAX_MISSING,
             min_coverage: float = ALMOST_MIN_COVERAGE) -> str:
    if required_total <= 0:
        raise ValueError("A build must require at least one piece")
    if missing_total == 0:
        return STATUS_BUILDABLE
    coverage = (required_total - missing_total) / required_total
    if missing_total <= max_missing and coverage >= min_coverage:
        return STATUS_ALMOST
    return STATUS_NEEDS_PARTS


@dataclass
class BuildCheck:
    # One row per required line: part_num, color_id, required, owned (exact pieces
    # you have), available, substituted (of available, how many are substitutes),
    # missing, substitutes (list of (part_num, color_id, qty) used for this line)
    lines: pd.DataFrame
    # The physical pieces that would be used: part_num, color_id, quantity
    holds: pd.DataFrame
    required_total: int
    available_total: int
    missing_total: int
    substituted_total: int
    coverage: float
    status: str

    @property
    def missing_lines(self) -> pd.DataFrame:
        return self.lines[self.lines["missing"] > 0].reset_index(drop=True)


def allocate(required: pd.DataFrame, inventory: pd.DataFrame,
             rules: MatchRules = EXACT) -> BuildCheck:
    """
    Decide which of your pieces would cover each line of ONE build.

    required  : columns part_num, color_id, quantity
    inventory : columns part_num, color_id, quantity (the pieces free to use)
    """
    columns = KEY + ["quantity"]
    req = normalize_pieces(required[columns] if not required.empty else required)
    if req.empty:
        raise ValueError("A build must require at least one piece")
    inv = normalize_pieces(inventory[columns] if not inventory.empty else inventory)
    free = {(p, c): q for p, c, q in inv.itertuples(index=False, name=None)}
    owned_exact = dict(free)
    used = defaultdict(int)

    lines = []
    # Pass 1: exact pieces.
    for part, colour, need in req.itertuples(index=False, name=None):
        take = min(need, free.get((part, colour), 0))
        if take:
            free[(part, colour)] -= take
            used[(part, colour)] += take
        lines.append({"part_num": part, "color_id": colour, "required": need,
                      "owned": owned_exact.get((part, colour), 0),
                      "available": take, "substituted": 0, "substitutes": []})

    # Pass 2: substitutes. Owned pieces are grouped by their key; each line
    # looks in the group its NEEDED key points to.
    groups = defaultdict(list)
    for key in free:
        groups[(rules.part_key(key[0]), rules.color_key(key[1]))].append(key)
    for line in lines:
        still = line["required"] - line["available"]
        if still == 0:
            continue
        part, colour = line["part_num"], line["color_id"]
        candidates = sorted(
            (k for k in groups[(rules.need_part_key(part), rules.color_key(colour))]
             if k != (part, colour) and free[k] > 0),
            key=lambda k: (k[0] != part, k[1] != colour, -free[k], k))
        for k in candidates:
            if still == 0:
                break
            take = min(still, free[k])
            free[k] -= take
            used[k] += take
            still -= take
            line["available"] += take
            line["substituted"] += take
            line["substitutes"].append((k[0], k[1], take))

    lines = pd.DataFrame(lines)
    lines["missing"] = lines["required"] - lines["available"]
    holds = pd.DataFrame([(p, c, q) for (p, c), q in used.items() if q > 0],
                         columns=["part_num", "color_id", "quantity"])

    required_total = int(lines["required"].sum())
    available_total = int(lines["available"].sum())
    missing_total = required_total - available_total
    return BuildCheck(
        lines=lines[["part_num", "color_id", "required", "owned", "available",
                     "substituted", "missing", "substitutes"]],
        holds=holds,
        required_total=required_total,
        available_total=available_total,
        missing_total=missing_total,
        substituted_total=int(lines["substituted"].sum()),
        coverage=available_total / required_total,
        status=classify(required_total, missing_total),
    )


def check_build(required: pd.DataFrame, inventory: pd.DataFrame,
                rules: MatchRules = EXACT) -> BuildCheck:
    """Detailed, line-by-line check of ONE build against your pieces."""
    return allocate(required, inventory, rules)


def group_lines(build_parts: pd.DataFrame, inventory: pd.DataFrame,
                rules: MatchRules = EXACT) -> pd.DataFrame:
    """
    Every build's needs per NEEDED group, compared with what you own.

    Two stages, the same as allocate() but for all builds at once:
      1. exact:       each line uses min(required, owned) of its exact piece
      2. substitutes: what's still needed in a group is covered from the owned
                      pieces of that group that stage 1 didn't use for this build

    Returns: build_id, part_key, color_key, required, available, substituted, missing
    """
    bp = normalize_pieces(build_parts).rename(columns={"quantity": "required"})
    inv = normalize_pieces(inventory)

    # Stage 1: exact pieces.
    lines = bp.merge(inv.rename(columns={"quantity": "owned"}), on=KEY, how="left")
    lines["owned"] = lines["owned"].fillna(0).astype(int)
    lines["exact"] = lines[["required", "owned"]].min(axis=1)
    lines["leftover"] = lines["required"] - lines["exact"]

    need_keys = add_group_keys(lines[KEY], rules, needed=True)
    own_keys = add_group_keys(lines[KEY], rules)
    lines["part_key"], lines["color_key"] = need_keys["part_key"], need_keys["color_key"]

    # Stage 2: the substitute pool per group = everything owned in the group,
    # minus what stage 1 already used of those pieces for this build.
    pool = add_group_keys(inv, rules).groupby(GROUP_KEY)["quantity"].sum().rename("pool")
    used = (lines.assign(part_key=own_keys["part_key"], color_key=own_keys["color_key"])
            .groupby(["build_id"] + GROUP_KEY)["exact"].sum().rename("used"))

    out = lines.groupby(["build_id"] + GROUP_KEY).agg(
        required=("required", "sum"), exact=("exact", "sum"), leftover=("leftover", "sum"))
    out = out.join(pool, on=GROUP_KEY).join(used)
    free_pool = (out["pool"].fillna(0) - out["used"].fillna(0)).clip(lower=0)
    out["substituted"] = pd.concat([out["leftover"], free_pool], axis=1).min(axis=1).astype(int)
    out["available"] = out["exact"] + out["substituted"]
    out["missing"] = out["required"] - out["available"]
    return out.reset_index()[["build_id", "part_key", "color_key", "required", "available",
                              "substituted", "missing"]]


def score_builds(build_parts: pd.DataFrame, inventory: pd.DataFrame,
                 rules: MatchRules = EXACT) -> pd.DataFrame:
    """
    The same totals as check_build, for EVERY build at once.

    Looping allocate() over thousands of builds would be slow. Instead,
    group_lines() does the same two stages (exact, then substitutes per group)
    with a few big merges and group-bys, reaching exactly allocate()'s totals.

    Returns one row per build:
        build_id, required_total, available_total, missing_total,
        missing_lines (how many distinct pieces/groups are short), coverage, status
    """
    lines = group_lines(build_parts, inventory, rules)
    if lines.empty:
        return pd.DataFrame(columns=["build_id", "required_total", "available_total",
                                     "missing_total", "missing_lines", "coverage", "status"])
    lines["is_short"] = (lines["missing"] > 0).astype(int)

    scores = lines.groupby("build_id", as_index=False).agg(
        required_total=("required", "sum"),
        available_total=("available", "sum"),
        missing_total=("missing", "sum"),
        missing_lines=("is_short", "sum"),
    )
    scores["coverage"] = scores["available_total"] / scores["required_total"]
    scores["status"] = [classify(r, m) for r, m in
                        zip(scores["required_total"], scores["missing_total"])]
    return scores

"""
substitutions.py — Which pieces can stand in for each other.

Part substitutions come from Rebrickable's part_relationships data, which
links pairs of parts with a type. Only the types where one part really can
replace the other are offered, each as a separate opt-in:

  M  mold variant   the same piece made in a different factory mold
                    (e.g. hollow vs solid studs).                    on by default
                    Two-way. A few odd ones exist: Rebrickable lists the
                    Bionicle Krana masks as molds of each other.
  A  alternate      parts Rebrickable lists as acceptable stand-ins.
                    Two-way. Mostly fine, but looser: e.g. a 2x4
                    magnet brick counts as an alternate 2x4 brick.  off by default
  P  print,         printed or marbled version of a plain part.
  T  pattern        ONE-WAY: the plain part may cover a printed one
                    (you build it without the print). A printed part
                    never replaces another print: all printed heads
                    link to the same plain head, and two-way grouping
                    would let any face stand in for any other.       off by default

Never used, because the parts are NOT interchangeable:
  R  pair           parts that go together (a tyre and its wheel)
  B  sub-part       one part of a bigger assembly

Two-way links chain together: if A~B and B~C then A, B and C form one group.
Grouping uses union-find: each part points to a "parent", and following
parents leads to the group's representative.

Colour substitutions:
  exact    colours must match
  similar  LEGO's 2003-04 colour change: old Light Gray ~ Light Bluish Gray,
           Dark Gray ~ Dark Bluish Gray, Brown ~ Reddish Brown. Models from
           before and after the change used them for the same purpose.
  any      colour is ignored entirely
"""

from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from buildfinder.compatibility import MatchRules
from buildfinder.database import connect, get_db_path

TWO_WAY_OPTIONS = {
    "molds": ("M",),
    "alternates": ("A",),
}
PRINT_TYPES = ("P", "T")         # one-way: printed child -> plain parent
NEVER_SUBSTITUTE = ("R", "B")

COLOR_MODES = ("exact", "similar", "any")
SIMILAR_COLOR_GROUPS = [(7, 71), (8, 72), (6, 70)]   # (old, new) colour ids


def union_find_groups(pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """
    Group items linked by pairs. Returns {item: representative} for every
    item that appears in a pair. The representative is the group's
    alphabetically smallest member, so results are stable.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path halving: keeps chains short
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Smaller id becomes the root, so the root is the group minimum.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra
    return {x: find(x) for x in parent}


@lru_cache(maxsize=16)
def _part_groups(db_path: str, rel_types: Tuple[str, ...]) -> Dict[str, str]:
    # db_path is part of the cache key so tests (temp databases) don't share results.
    if not rel_types:
        return {}
    placeholders = ",".join("?" for _ in rel_types)
    with connect() as conn:
        pairs = conn.execute(
            f"SELECT child_part_num, parent_part_num FROM part_relationships "
            f"WHERE rel_type IN ({placeholders})", rel_types).fetchall()
    return union_find_groups(pairs)


def part_groups(molds: bool = True, alternates: bool = False) -> Dict[str, str]:
    """Two-way groups: {part: group representative}."""
    chosen = {"molds": molds, "alternates": alternates}
    rel_types = tuple(sorted(t for name, on in chosen.items() if on for t in TWO_WAY_OPTIONS[name]))
    return _part_groups(str(get_db_path()), rel_types)


@lru_cache(maxsize=4)
def _plain_versions(db_path: str) -> Dict[str, str]:
    with connect() as conn:
        pairs = conn.execute(
            "SELECT child_part_num, MIN(parent_part_num) FROM part_relationships "
            "WHERE rel_type IN ('P', 'T') GROUP BY child_part_num").fetchall()
    parent = dict(pairs)
    # Follow chains (a print of a print) to the plain root; the step limit
    # guards against a cycle in the data.
    resolved = {}
    for child in parent:
        node, steps = child, 0
        while node in parent and steps < 10:
            node, steps = parent[node], steps + 1
        if node != child:
            resolved[child] = node
    return resolved


def plain_versions() -> Dict[str, str]:
    """One-way: {printed/patterned part: its plain version}."""
    return _plain_versions(str(get_db_path()))


def color_groups(mode: str) -> Dict[int, int]:
    if mode not in COLOR_MODES:
        raise ValueError(f"colour mode must be one of {COLOR_MODES}")
    if mode != "similar":
        return {}
    return {c: group[0] for group in SIMILAR_COLOR_GROUPS for c in group}


def build_rules(color_mode: str = "exact", molds: bool = True, alternates: bool = False,
                prints: bool = False) -> MatchRules:
    return MatchRules(
        part_map=part_groups(molds, alternates),
        color_map=color_groups(color_mode),
        any_color=(color_mode == "any"),
        need_part_map=plain_versions() if prints else {},
    )


def equivalent_parts(part_num: str, rules: MatchRules) -> List[str]:
    """Every part you could use where a build needs part_num (itself included)."""
    key = rules.need_part_key(part_num)
    return sorted({p for p, k in rules.part_map.items() if k == key} | {part_num, key})


def clear_cache() -> None:
    _part_groups.cache_clear()
    _plain_versions.cache_clear()

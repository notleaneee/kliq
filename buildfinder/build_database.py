"""
build_database.py — Create and read the database of candidate builds.

Stage 1 builds are official LEGO sets, because Rebrickable publishes their
complete parts lists. For each set we compute its "requirements": the exact
(part_num, color_id, quantity) rows you need to build it.

Rules for the requirements:
  * Spare parts are excluded (LEGO adds extra small pieces to the bag).
  * Stickers are excluded (you can build without them, and nobody stores loose stickers).
  * Minifigure parts ARE included: a minifig's own parts list is expanded and
    multiplied by how many of that minifig the set contains.
  * If a set has several inventory versions (corrections, regional variants),
    version 1 is used.
  * Sets that contain other sets (multi-packs, bundles) are skipped.

Not available from this data, and shown honestly as "not available" in the app:
difficulty, build time, and verified instructions. Each build links to its
Rebrickable page, which usually links to the official LEGO instructions, but
that is not checked here.
"""

from typing import Dict, Optional

import pandas as pd

from buildfinder.database import connect, init_db, CATALOG_TABLES

STICKER_CATEGORY_ID = 58
MAX_MULTIPACK_DEPTH = 5

EXCLUDED_THEMES = {
    "Bulk Bricks", "Service Packs", "Activity Books with LEGO Parts", "Ideas Books",
    "Story Books", "Gear", "Key Chain", "Magnets",
}

SOURCE_OFFICIAL = "Rebrickable sets data (official LEGO set)"
INSTRUCTIONS_NOTE = ("Link goes to the Rebrickable set page, which usually links to the "
                     "official LEGO instructions. Availability has not been verified.")


# ── Pure functions (no database) ──────────────────────────────────────────────

def _first_version_inventories(inventories: pd.DataFrame) -> pd.DataFrame:
    """One inventory per set_num: the one with the lowest version number."""
    return (inventories.sort_values(["set_num", "version"])
            .drop_duplicates("set_num", keep="first")[["id", "set_num"]]
            .rename(columns={"id": "inventory_id"}))


_SET_PARTS_KEY = ["set_num", "part_num", "color_id", "is_spare"]


def _group(df: pd.DataFrame, key) -> pd.DataFrame:
    out = df.groupby(key, as_index=False)["quantity"].sum()
    out["color_id"] = out["color_id"].astype(int)
    out["quantity"] = out["quantity"].astype(int)
    return out


def compute_set_parts(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Everything physically in each set's box (stickers aside).

    Returns columns: set_num, part_num, color_id, is_spare (0/1), quantity
    with one row per (set_num, part_num, color_id, is_spare).

      * minifigs are expanded into their parts (x how many of each fig)
      * multi-packs are expanded into the sets they contain (nested ones too)
      * spare parts are kept but flagged, so callers can choose
    """
    inv = _first_version_inventories(raw["inventories"])

    rows = raw["inventory_parts"].copy()
    rows["is_spare"] = rows["is_spare"].astype(str).str.lower().isin(["true", "t", "1"]).astype(int)
    sticker_parts = set(raw["parts"].loc[raw["parts"]["part_cat_id"] == STICKER_CATEGORY_ID,
                                         "part_num"])
    rows = rows[~rows["part_num"].isin(sticker_parts)]

    # Direct parts of every chosen inventory (sets and minifigs alike).
    direct = rows.merge(inv, on="inventory_id")[_SET_PARTS_KEY + ["quantity"]]

    # Minifig expansion: set inventory -> fig_num (x qty) -> the fig's own parts.
    figs = raw["inventory_minifigs"].merge(inv, on="inventory_id")[["set_num", "fig_num", "quantity"]]
    figs = figs.rename(columns={"quantity": "fig_qty"})
    expanded = figs.merge(direct.rename(columns={"set_num": "fig_num"}), on="fig_num")
    expanded["quantity"] = expanded["quantity"] * expanded["fig_qty"]
    with_figs = _group(pd.concat([direct, expanded[_SET_PARTS_KEY + ["quantity"]]]), _SET_PARTS_KEY)

    # Multi-pack expansion: parent inventory -> child set (x qty) -> child's parts.
    # A few multi-packs contain other multi-packs, so repeat until nothing
    # changes: each round, every set = its own parts + its children's full parts.
    subs = raw["inventory_sets"].rename(columns={"set_num": "child_set", "quantity": "sub_qty"})
    subs = subs.merge(inv, on="inventory_id")[["set_num", "child_set", "sub_qty"]]
    full = with_figs
    for _ in range(MAX_MULTIPACK_DEPTH):
        children = subs.merge(full.rename(columns={"set_num": "child_set"}), on="child_set")
        children["quantity"] = children["quantity"] * children["sub_qty"]
        expanded = _group(pd.concat([with_figs, children[_SET_PARTS_KEY + ["quantity"]]]),
                          _SET_PARTS_KEY)
        if len(expanded) == len(full) and expanded["quantity"].sum() == full["quantity"].sum():
            break
        full = expanded
    return full


def requirements_from_set_parts(set_parts: pd.DataFrame) -> pd.DataFrame:
    """What you need to BUILD a set: everything except the spares."""
    non_spare = set_parts[set_parts["is_spare"] == 0]
    return _group(non_spare, ["set_num", "part_num", "color_id"])


def compute_requirements(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Parts needed to build every set (and minifig).

    Returns columns: set_num, part_num, color_id, quantity
    with exactly one row per (set_num, part_num, color_id).
    """
    return requirements_from_set_parts(compute_set_parts(raw))


def select_builds(raw: Dict[str, pd.DataFrame], requirements: pd.DataFrame,
                  min_parts: int = 20, max_parts: int = 400) -> pd.DataFrame:
    """
    Choose which sets become candidate builds, with their metadata.

    Returns columns matching the `builds` table.
    """
    sets = raw["sets"].merge(
        raw["themes"][["id", "name"]].rename(columns={"id": "theme_id", "name": "theme"}),
        on="theme_id", how="left",
    )
    sets = sets[~sets["theme"].isin(EXCLUDED_THEMES)]

    # Multi-pack sets: their inventory lists other sets instead of parts.
    inv = _first_version_inventories(raw["inventories"])
    parents = inv[inv["inventory_id"].isin(raw["inventory_sets"]["inventory_id"])]["set_num"]
    sets = sets[~sets["set_num"].isin(parents)]

    totals = requirements.groupby("set_num", as_index=False)["quantity"].sum()
    totals = totals.rename(columns={"quantity": "num_parts_required"})
    sets = sets.merge(totals, on="set_num")      # inner join drops sets with no parts list
    sets = sets[sets["num_parts_required"].between(min_parts, max_parts)]

    return pd.DataFrame({
        "build_id": sets["set_num"],
        "name": sets["name"],
        "year": sets["year"].astype("Int64"),
        "theme": sets["theme"].fillna("Unknown"),
        "num_parts": sets["num_parts_required"].astype(int),
        "img_url": sets["img_url"],
        "instructions_url": "https://rebrickable.com/sets/" + sets["set_num"] + "/",
        "instructions_note": INSTRUCTIONS_NOTE,
        "difficulty": None,
        "build_time_minutes": None,
        "source": SOURCE_OFFICIAL,
    }).reset_index(drop=True)


def compute_part_colors(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Every (part_num, color_id) that appears in some inventory, with one image
    URL and num_sets: how many inventories (sets/minifigs) contain it.
    """
    rows = raw["inventory_parts"]
    pc = rows.groupby(["part_num", "color_id"], as_index=False).agg(
        img_url=("img_url", "first"),            # "first" skips missing images
        num_sets=("inventory_id", "nunique"))
    pc["color_id"] = pc["color_id"].astype(int)
    return pc


def compute_parts(raw: Dict[str, pd.DataFrame], part_colors: pd.DataFrame) -> pd.DataFrame:
    """
    Parts with a picture (from their most common colour) and num_sets, so the
    piece picker can show pictures and list common pieces first.
    """
    counts = raw["inventory_parts"].groupby("part_num")["inventory_id"].nunique().rename("num_sets")
    best_image = (part_colors.dropna(subset=["img_url"])
                  .sort_values(["part_num", "num_sets"], ascending=[True, False])
                  .drop_duplicates("part_num").set_index("part_num")["img_url"])
    parts = raw["parts"][["part_num", "name", "part_cat_id"]].copy()
    parts["img_url"] = parts["part_num"].map(best_image)
    parts["num_sets"] = parts["part_num"].map(counts).fillna(0).astype(int)
    return parts


# ── Writing the catalog ──────────────────────────────────────────────────────

def write_catalog(raw: Dict[str, pd.DataFrame], min_parts: int = 20, max_parts: int = 400) -> dict:
    """Rebuild every catalog table. Your inventory and projects are left untouched."""
    print("Computing parts lists...")
    set_parts = compute_set_parts(raw)
    requirements = requirements_from_set_parts(set_parts)
    builds = select_builds(raw, requirements, min_parts, max_parts)
    build_parts = (requirements[requirements["set_num"].isin(builds["build_id"])]
                   .rename(columns={"set_num": "build_id"}))

    sets = raw["sets"].merge(
        raw["themes"][["id", "name"]].rename(columns={"id": "theme_id", "name": "theme"}),
        on="theme_id", how="left")[["set_num", "name", "year", "theme", "num_parts", "img_url"]]
    # Only real sets are importable; minifig inventories ("fig-...") are not.
    set_parts = set_parts[set_parts["set_num"].isin(sets["set_num"])]

    part_colors = compute_part_colors(raw)
    tables = {
        "colors": raw["colors"][["id", "name", "rgb", "is_trans"]].assign(
            is_trans=lambda d: d["is_trans"].astype(str).str.lower().isin(["true", "t", "1"]).astype(int)),
        "part_categories": raw["part_categories"][["id", "name"]],
        "parts": compute_parts(raw, part_colors),
        "part_colors": part_colors,
        "builds": builds,
        "build_parts": build_parts[["build_id", "part_num", "color_id", "quantity"]],
        "sets": sets,
        "set_parts": set_parts[_SET_PARTS_KEY + ["quantity"]],
        "part_relationships": raw["part_relationships"][
            ["rel_type", "child_part_num", "parent_part_num"]],
    }

    init_db()
    with connect() as conn:
        # One transaction: if anything fails, the old catalog stays intact.
        for name in CATALOG_TABLES:
            conn.execute(f"DELETE FROM {name}")
            tables[name].to_sql(name, conn, if_exists="append", index=False, chunksize=50_000)

    counts = {name: len(df) for name, df in tables.items()}
    for name, n in counts.items():
        print(f"  {name:<16} {n:>10,} rows")
    return counts


# ── Reading the catalog ──────────────────────────────────────────────────────

def get_builds() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM builds", conn)


def get_all_build_parts() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT build_id, part_num, color_id, quantity FROM build_parts", conn)


def get_build_parts(build_id: str) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT part_num, color_id, quantity FROM build_parts WHERE build_id = ?",
            conn, params=(build_id,))


def describe_pieces(pieces: pd.DataFrame) -> pd.DataFrame:
    """
    Add human-readable columns (part_name, color_name, img_url) to any frame
    that has part_num and color_id columns.
    """
    if pieces.empty:
        return pieces.assign(part_name=[], color_name=[], img_url=[])
    wanted = pieces[["part_num", "color_id"]].drop_duplicates()
    with connect() as conn:
        # A temporary table lets SQLite join only the pieces we need,
        # instead of loading all ~65,000 parts into pandas.
        conn.execute("CREATE TEMP TABLE wanted (part_num TEXT, color_id INTEGER)")
        # sqlite3 can't bind numpy integers, so convert to plain Python types.
        conn.executemany("INSERT INTO wanted VALUES (?, ?)",
                         [(str(p), int(c)) for p, c in wanted.itertuples(index=False, name=None)])
        names = pd.read_sql_query(
            """
            SELECT w.part_num, w.color_id, p.name AS part_name,
                   c.name AS color_name, pc.img_url
            FROM wanted w
            LEFT JOIN parts p        ON p.part_num = w.part_num
            LEFT JOIN colors c       ON c.id = w.color_id
            LEFT JOIN part_colors pc ON pc.part_num = w.part_num AND pc.color_id = w.color_id
            """,
            conn,
        )
    return pieces.merge(names, on=["part_num", "color_id"], how="left")


def get_build(build_id: str) -> Optional[dict]:
    with connect() as conn:
        df = pd.read_sql_query("SELECT * FROM builds WHERE build_id = ?", conn, params=(build_id,))
    return None if df.empty else df.iloc[0].to_dict()

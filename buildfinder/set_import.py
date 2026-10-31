"""
set_import.py — Add every piece of an official set you own to your inventory.

Real-life complications this handles:

  * Copies: you own the same set twice.
  * Spare parts: LEGO includes extras of small pieces. They're real pieces you
    probably have, so they're included by default (you can turn that off).
  * Pieces missing from your copy: you lost a few. Pass them as `missing`
    and they're subtracted before anything is added.
  * The set is still built on your shelf: those pieces are yours but in use.
    With currently_built=True the pieces are added AND a "built" project is
    created that holds them, so they aren't recommended for other builds
    until you disassemble it.

Everything happens in one database transaction: either the whole import
succeeds, or nothing changes.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from buildfinder.compatibility import normalize_pieces
from buildfinder.database import connect
from buildfinder.inventory import upsert_pieces
from buildfinder.projects import BUILT, start_project

MAX_COPIES = 100


class SetImportError(ValueError):
    pass


@dataclass
class SetImportReport:
    set_num: str
    name: str
    copies: int
    pieces_added: int
    pieces_missing_excluded: int
    project_id: Optional[int] = None


def sets_loaded() -> bool:
    """False if the catalog was built before set import existed (re-run build_finder_catalog.py)."""
    with connect() as conn:
        return conn.execute("SELECT EXISTS (SELECT 1 FROM sets)").fetchone()[0] == 1


def normalize_set_num(text: str) -> str:
    """'30669' -> '30669-1'. Rebrickable set numbers end in a variant suffix."""
    text = (text or "").strip()
    if not text:
        raise SetImportError("enter a set number")
    return text if "-" in text else f"{text}-1"


def find_sets(query: str, limit: int = 30) -> pd.DataFrame:
    """Search sets by number (with or without '-1') or by name words."""
    query = (query or "").strip()
    if not query:
        return pd.DataFrame(columns=["set_num", "name", "year", "theme", "num_parts", "img_url"])
    words = query.split()
    name_clause = " AND ".join("name LIKE ?" for _ in words)
    with connect() as conn:
        return pd.read_sql_query(
            f"""
            SELECT set_num, name, year, theme, num_parts, img_url FROM sets
            WHERE set_num = ? OR set_num = ? OR ({name_clause})
            ORDER BY (set_num = ? OR set_num = ?) DESC, year DESC, name
            LIMIT ?
            """,
            conn,
            params=[query, f"{query}-1"] + [f"%{w}%" for w in words] + [query, f"{query}-1", limit],
        )


def get_set(set_num: str) -> Optional[dict]:
    with connect() as conn:
        df = pd.read_sql_query("SELECT * FROM sets WHERE set_num = ?", conn, params=(set_num,))
    return None if df.empty else df.iloc[0].to_dict()


def set_contents(set_num: str, include_spares: bool = True) -> pd.DataFrame:
    """What's in one box: (part_num, color_id, quantity), spares optional."""
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT part_num, color_id, is_spare, quantity FROM set_parts WHERE set_num = ?",
            conn, params=(set_num,))
    if not include_spares:
        df = df[df["is_spare"] == 0]
    return normalize_pieces(df)


def import_set(set_num: str, copies: int = 1, include_spares: bool = True,
               missing: Optional[pd.DataFrame] = None,
               currently_built: bool = False) -> SetImportReport:
    """
    Add a set's pieces to your inventory.

    missing         : optional (part_num, color_id, quantity) rows you DON'T have.
                      Each must be a piece of this set, at most the quantity in it.
    currently_built : also create a "built" project holding the pieces of the
                      model (all non-spare pieces), so they count as in use.
    """
    set_num = normalize_set_num(set_num)
    info = get_set(set_num)
    if info is None:
        raise SetImportError(f"unknown set '{set_num}'")
    if not isinstance(copies, int) or not 1 <= copies <= MAX_COPIES:
        raise SetImportError(f"copies must be a whole number from 1 to {MAX_COPIES}")

    contents = set_contents(set_num, include_spares)
    if contents.empty:
        raise SetImportError(f"Rebrickable has no parts list for {set_num}")
    contents["quantity"] *= copies

    missing_total = 0
    if missing is not None and not missing.empty:
        lost = normalize_pieces(missing).rename(columns={"quantity": "lost"})
        check = lost.merge(contents, on=["part_num", "color_id"], how="left")
        bad = check[check["quantity"].isna() | (check["lost"] > check["quantity"])]
        if not bad.empty:
            r = bad.iloc[0]
            raise SetImportError(
                f"missing piece {r.part_num} (colour {r.color_id}) x{r.lost} isn't in "
                f"{copies} x {set_num} in that quantity")
        contents = contents.merge(lost, on=["part_num", "color_id"], how="left")
        contents["lost"] = contents["lost"].fillna(0).astype(int)
        contents["quantity"] -= contents["lost"]
        missing_total = int(contents["lost"].sum())
        contents = contents[contents["quantity"] > 0][["part_num", "color_id", "quantity"]]

    project_id = None
    with connect() as conn:
        upsert_pieces(conn, contents.itertuples(index=False, name=None), notes=f"from set {set_num}")
        if currently_built:
            # The model needs the non-spare pieces. Holding is partial: if a
            # piece is lost and no spare can stand in, the project simply
            # shows that piece as missing, which is the truth.
            model = set_contents(set_num, include_spares=False)
            model["quantity"] *= copies
            name = info["name"] + (f" (x{copies})" if copies > 1 else "")
            project_id = start_project(conn, set_num, name, model, BUILT,
                                       allow_partial=True, notes="imported as built").project_id

    return SetImportReport(set_num, info["name"], copies, int(contents["quantity"].sum()),
                           missing_total, project_id)

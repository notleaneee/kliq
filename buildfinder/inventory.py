"""
inventory.py — Your LEGO piece inventory: add, edit, remove, search, CSV import/export.

Every piece is identified by (part_num, color_id). Adding a piece you already
own increases its quantity. It never creates a second row. This is enforced
twice: in code (the "upsert" below) and by the table's PRIMARY KEY.

All user input is validated here before it reaches the database:
  * the part number must exist in the Rebrickable parts list
  * the colour must exist
  * the quantity must be a whole number from 1 to MAX_QUANTITY

`quantity` is how many you OWN. Some of those may be held by projects
(reserved or built, see projects.py). You can't lower a quantity below what
your projects are holding: release the project first.
"""

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Union

import pandas as pd

from buildfinder.build_database import STICKER_CATEGORY_ID
from buildfinder.database import HELD_PIECES_SQL, connect

MAX_QUANTITY = 100_000

_UPSERT_SQL = """
INSERT INTO inventory (part_num, color_id, quantity, notes, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (part_num, color_id) DO UPDATE SET
    quantity   = inventory.quantity + excluded.quantity,
    notes      = CASE
                   WHEN excluded.notes IS NULL OR excluded.notes = '' THEN inventory.notes
                   WHEN inventory.notes IS NULL OR inventory.notes = '' THEN excluded.notes
                   WHEN instr(inventory.notes, excluded.notes) > 0 THEN inventory.notes
                   ELSE inventory.notes || '; ' || excluded.notes
                 END,
    updated_at = excluded.updated_at
"""


class InventoryError(ValueError):
    """Raised for invalid user input (unknown part, bad quantity, ...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_notes(notes: Optional[str]) -> Optional[str]:
    # Empty notes arrive as None, "" or NaN depending on the source. NaN is
    # sneaky: NaN != NaN, so it must be caught before any comparison.
    if notes is None or (isinstance(notes, float) and pd.isna(notes)):
        return None
    notes = str(notes).strip()
    return notes or None


def _parse_quantity(value) -> int:
    """Accept 5, "5", "5.0" — reject 0, negatives, 2.5, "abc"."""
    try:
        number = float(str(value).strip())
    except ValueError:
        raise InventoryError(f"quantity '{value}' is not a number")
    if not number.is_integer():
        raise InventoryError(f"quantity '{value}' must be a whole number")
    number = int(number)
    if not 1 <= number <= MAX_QUANTITY:
        raise InventoryError(f"quantity must be between 1 and {MAX_QUANTITY:,}, got {number}")
    return number


def _check_piece(conn, part_num: str, color_id: int) -> None:
    if not conn.execute("SELECT 1 FROM parts WHERE part_num = ?", (part_num,)).fetchone():
        raise InventoryError(f"unknown part number '{part_num}'")
    if not conn.execute("SELECT 1 FROM colors WHERE id = ?", (color_id,)).fetchone():
        raise InventoryError(f"unknown colour id {color_id}")


def _held(conn, part_num: str, color_id: int) -> int:
    row = conn.execute(
        f"SELECT held FROM ({HELD_PIECES_SQL}) WHERE part_num = ? AND color_id = ?",
        (str(part_num), int(color_id))).fetchone()
    return int(row[0]) if row else 0


def _check_not_below_held(conn, part_num: str, color_id: int, new_quantity: int) -> None:
    held = _held(conn, part_num, color_id)
    if new_quantity < held:
        raise InventoryError(
            f"{held} of part {part_num} (colour {color_id}) are held by your projects, so the "
            f"quantity can't go below {held}. Cancel or disassemble those projects first.")


def upsert_pieces(conn, rows, notes: Optional[str] = None) -> None:
    """
    Add many (part_num, color_id, quantity) rows inside an existing transaction.
    Quantities are summed with what you already own. Rows must already be valid.
    """
    now = _now()
    conn.executemany(_UPSERT_SQL, [(str(p), int(c), int(q), _clean_notes(notes), now)
                                   for p, c, q in rows])


# ── Single-piece operations ──────────────────────────────────────────────────

def add_piece(part_num: str, color_id: int, quantity, notes: Optional[str] = None) -> None:
    """Add pieces. If you already own this part in this colour, quantities are summed."""
    part_num = str(part_num).strip()
    color_id = int(color_id)
    quantity = _parse_quantity(quantity)
    with connect() as conn:
        _check_piece(conn, part_num, color_id)
        conn.execute(_UPSERT_SQL, (part_num, color_id, quantity, _clean_notes(notes), _now()))


def set_quantity(part_num: str, color_id: int, quantity) -> None:
    """Overwrite the owned quantity. Setting it to 0 removes the piece."""
    if str(quantity).strip() in ("0", "0.0"):
        remove_piece(part_num, color_id)
        return
    quantity = _parse_quantity(quantity)
    with connect() as conn:
        _check_not_below_held(conn, part_num, color_id, quantity)
        cur = conn.execute(
            "UPDATE inventory SET quantity = ?, updated_at = ? WHERE part_num = ? AND color_id = ?",
            (quantity, _now(), str(part_num), int(color_id)))
        if cur.rowcount == 0:
            raise InventoryError(f"you don't own part {part_num} in colour {color_id}")


def set_notes(part_num: str, color_id: int, notes: Optional[str]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE inventory SET notes = ?, updated_at = ? WHERE part_num = ? AND color_id = ?",
            (_clean_notes(notes), _now(), str(part_num), int(color_id)))


def remove_piece(part_num: str, color_id: int) -> None:
    with connect() as conn:
        _check_not_below_held(conn, part_num, color_id, 0)
        conn.execute("DELETE FROM inventory WHERE part_num = ? AND color_id = ?",
                     (str(part_num), int(color_id)))


def clear_inventory() -> None:
    """Delete every piece AND every project (a project can't hold pieces you don't own)."""
    with connect() as conn:
        conn.execute("DELETE FROM project_holds")
        conn.execute("DELETE FROM project_parts")
        conn.execute("DELETE FROM projects")
        conn.execute("DELETE FROM inventory")


def apply_table_edits(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """
    Save edits made in the app's inventory table. Returns how many changes were made.

    before : rows as shown (part_num, color_id, quantity, notes)
    after  : the same rows after editing, same order, plus a boolean `remove` column.
             A quantity of 0 (or blank) also removes the piece.

    All rows are checked before anything is saved, so an invalid edit (like
    going below what a project holds) never leaves the table half-saved.
    """
    if len(before) != len(after):
        raise InventoryError("edited table has a different number of rows")

    plan = []   # (action, part_num, color_id, value)
    for (_, old), (_, new) in zip(before.iterrows(), after.iterrows()):
        part_num, color_id = old["part_num"], int(old["color_id"])
        qty = new["quantity"]
        if bool(new.get("remove", False)) or pd.isna(qty) or int(qty) == 0:
            plan.append(("remove", part_num, color_id, 0))
            continue
        if int(qty) != int(old["quantity"]):
            plan.append(("quantity", part_num, color_id, _parse_quantity(qty)))
        if _clean_notes(new["notes"]) != _clean_notes(old["notes"]):
            plan.append(("notes", part_num, color_id, new["notes"]))

    with connect() as conn:
        for action, part_num, color_id, value in plan:
            if action in ("remove", "quantity"):
                _check_not_below_held(conn, part_num, color_id, value)

    for action, part_num, color_id, value in plan:
        if action == "remove":
            remove_piece(part_num, color_id)
        elif action == "quantity":
            set_quantity(part_num, color_id, value)
        else:
            set_notes(part_num, color_id, value)
    return len(plan)


# ── Reading the inventory ────────────────────────────────────────────────────

def get_inventory_pieces() -> pd.DataFrame:
    """Every piece you OWN as bare (part_num, color_id, quantity) rows."""
    with connect() as conn:
        return pd.read_sql_query("SELECT part_num, color_id, quantity FROM inventory", conn)


def get_held_pieces() -> pd.DataFrame:
    """Pieces held by active projects: (part_num, color_id, quantity)."""
    with connect() as conn:
        return pd.read_sql_query(HELD_PIECES_SQL, conn).rename(columns={"held": "quantity"})


def get_available_pieces() -> pd.DataFrame:
    """
    Pieces free for a NEW build: owned minus held by projects.
    This is what recommendations use, so one physical piece is never counted
    for two builds at the same time.
    """
    with connect() as conn:
        return pd.read_sql_query(
            f"""
            SELECT i.part_num, i.color_id, i.quantity - COALESCE(h.held, 0) AS quantity
            FROM inventory i
            LEFT JOIN ({HELD_PIECES_SQL}) h ON h.part_num = i.part_num AND h.color_id = i.color_id
            WHERE i.quantity - COALESCE(h.held, 0) > 0
            """,
            conn,
        )


def get_inventory() -> pd.DataFrame:
    """Inventory with names, category, colour, image, and owned/held/available counts."""
    with connect() as conn:
        return pd.read_sql_query(
            f"""
            SELECT i.part_num, p.name AS part_name, pcat.name AS category,
                   i.color_id, c.name AS color_name, i.quantity,
                   COALESCE(h.held, 0) AS held,
                   i.quantity - COALESCE(h.held, 0) AS available,
                   i.notes, pc.img_url, i.updated_at
            FROM inventory i
            LEFT JOIN parts p           ON p.part_num = i.part_num
            LEFT JOIN part_categories pcat ON pcat.id = p.part_cat_id
            LEFT JOIN colors c          ON c.id = i.color_id
            LEFT JOIN part_colors pc    ON pc.part_num = i.part_num AND pc.color_id = i.color_id
            LEFT JOIN ({HELD_PIECES_SQL}) h ON h.part_num = i.part_num AND h.color_id = i.color_id
            ORDER BY pcat.name, p.name, c.name
            """,
            conn,
        )


def inventory_totals() -> dict:
    with connect() as conn:
        distinct, total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(quantity), 0) FROM inventory").fetchone()
        (held,) = conn.execute(f"SELECT COALESCE(SUM(held), 0) FROM ({HELD_PIECES_SQL})").fetchone()
    return {"distinct_pieces": distinct, "total_pieces": total,
            "held_pieces": held, "available_pieces": total - held}


# ── Search (for the "add a piece" form) ──────────────────────────────────────

def search_parts(query: str = "", category_id: Optional[int] = None,
                 color_id: Optional[int] = None, limit: int = 50) -> pd.DataFrame:
    """
    Find parts by part number or name words, optionally filtered by category
    and by a colour the part is known to exist in.

    "brick 2 x 4" matches "Brick 2 x 4" — every word must appear in the name.
    An exact part number match is listed first.
    """
    query = (query or "").strip()
    where, params = [], []

    if query:
        words = query.split()
        name_clause = " AND ".join("p.name LIKE ?" for _ in words)
        where.append(f"(p.part_num = ? OR ({name_clause}))")
        params += [query] + [f"%{w}%" for w in words]
    if category_id is not None:
        where.append("p.part_cat_id = ?")
        params.append(int(category_id))
    if color_id is not None:
        where.append("EXISTS (SELECT 1 FROM part_colors pc "
                     "WHERE pc.part_num = p.part_num AND pc.color_id = ?)")
        params.append(int(color_id))

    sql = """
        SELECT p.part_num, p.name, pcat.name AS category
        FROM parts p LEFT JOIN part_categories pcat ON pcat.id = p.part_cat_id
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    # Exact part number first, then shorter (usually more generic) names.
    sql += " ORDER BY (p.part_num = ?) DESC, length(p.name), p.name LIMIT ?"
    params += [query, int(limit)]

    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def browse_parts(query: str = "", category_id: Optional[int] = None,
                 limit: int = 24) -> pd.DataFrame:
    """
    Parts for the picture picker: part_num, name, category, img_url, num_sets.

    Same matching as search_parts (part number, or every word in the name),
    but only parts that appear in at least one set, most common first, so
    browsing "Bricks" starts with the bricks people actually have.
    """
    query = (query or "").strip()
    # Stickers are hidden: builds never count them, so adding them does nothing.
    where, params = ["p.num_sets > 0", "COALESCE(p.part_cat_id, -1) != ?"], [STICKER_CATEGORY_ID]
    if query:
        words = query.split()
        name_clause = " AND ".join("p.name LIKE ?" for _ in words)
        where.append(f"(p.part_num = ? OR ({name_clause}))")
        params += [query] + [f"%{w}%" for w in words]
    if category_id is not None:
        where.append("p.part_cat_id = ?")
        params.append(int(category_id))
    sql = f"""
        SELECT p.part_num, p.name, pcat.name AS category, p.img_url, p.num_sets
        FROM parts p LEFT JOIN part_categories pcat ON pcat.id = p.part_cat_id
        WHERE {" AND ".join(where)}
        ORDER BY (p.part_num = ?) DESC, p.num_sets DESC, p.name
        LIMIT ?
    """
    params += [query, int(limit)]
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def part_color_options(part_num: str) -> pd.DataFrame:
    """
    Colours this part comes in, most common first, each with a picture of the
    part in that colour: color_id, color_name, rgb, img_url, num_sets.
    """
    with connect() as conn:
        return pd.read_sql_query(
            """
            SELECT pc.color_id, c.name AS color_name, c.rgb, pc.img_url, pc.num_sets
            FROM part_colors pc JOIN colors c ON c.id = pc.color_id
            WHERE pc.part_num = ?
            ORDER BY pc.num_sets DESC, c.name
            """,
            conn, params=(str(part_num),))


def get_part(part_num: str) -> Optional[dict]:
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT p.part_num, p.name, pcat.name AS category, p.img_url FROM parts p "
            "LEFT JOIN part_categories pcat ON pcat.id = p.part_cat_id WHERE p.part_num = ?",
            conn, params=(str(part_num),))
    return None if df.empty else df.iloc[0].to_dict()


def colors_for_part(part_num: str) -> pd.DataFrame:
    """Colours this part has appeared in, according to Rebrickable set inventories."""
    with connect() as conn:
        return pd.read_sql_query(
            """
            SELECT c.id, c.name FROM part_colors pc JOIN colors c ON c.id = pc.color_id
            WHERE pc.part_num = ? ORDER BY c.name
            """,
            conn, params=(str(part_num),))


def all_colors() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT id, name FROM colors WHERE id >= 0 ORDER BY name", conn)


def all_categories() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT id, name FROM part_categories ORDER BY name", conn)


# ── CSV import / export ──────────────────────────────────────────────────────

_COLUMN_ALIASES = {
    "part_num": {"part_num", "part", "part_id", "partnum", "part_number"},
    "color": {"color_id", "color", "colour", "colour_id"},
    "quantity": {"quantity", "qty", "count"},
    "notes": {"notes", "note"},
}


@dataclass
class ImportReport:
    rows_read: int = 0
    rows_imported: int = 0
    pieces_added: int = 0
    duplicate_rows_combined: int = 0
    errors: List[str] = field(default_factory=list)


def _resolve_columns(columns) -> dict:
    """Map our standard names to whatever the CSV actually calls them."""
    normalised = {c: c.strip().lower().replace(" ", "_") for c in columns}
    found = {}
    for standard, aliases in _COLUMN_ALIASES.items():
        for original, norm in normalised.items():
            if norm in aliases:
                found[standard] = original
                break
    missing = [c for c in ("part_num", "color", "quantity") if c not in found]
    if missing:
        raise InventoryError(
            f"CSV is missing column(s): {', '.join(missing)}. "
            "Expected headers like: part_num,color_id,quantity,notes")
    return found


def import_csv(source: Union[str, bytes, io.IOBase]) -> ImportReport:
    """
    Import an inventory CSV. Valid rows are added; invalid rows are reported, not guessed.

    Accepted headers (case-insensitive): part_num/part, color_id/color (id or
    colour name), quantity/qty, and optional notes. This is meant to match
    Rebrickable's "Part,Color,Quantity" parts-list export, though that hasn't
    been tested against a real export file.

    Rows for the same (part, colour), inside the file or already in your
    inventory, are combined by adding their quantities.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        df = pd.read_csv(source, dtype=str, keep_default_na=False, skipinitialspace=True)
    except Exception as e:
        raise InventoryError(f"could not read CSV: {e}")

    cols = _resolve_columns(df.columns)
    report = ImportReport(rows_read=len(df))

    with connect() as conn:
        color_ids = {row[0] for row in conn.execute("SELECT id FROM colors")}
        color_names = {name.lower(): cid for cid, name in conn.execute("SELECT id, name FROM colors")}
        known_parts = {row[0] for row in conn.execute("SELECT part_num FROM parts")}

        combined = {}   # (part_num, color_id) -> [quantity, notes]
        for i, row in df.iterrows():
            line = i + 2  # +1 for the header row, +1 because spreadsheets count from 1
            part_num = row[cols["part_num"]].strip()
            raw_color = row[cols["color"]].strip()
            try:
                if part_num not in known_parts:
                    raise InventoryError(f"unknown part number '{part_num}'")
                if raw_color.lstrip("-").isdigit() and int(raw_color) in color_ids:
                    color_id = int(raw_color)
                elif raw_color.lower() in color_names:
                    color_id = color_names[raw_color.lower()]
                else:
                    raise InventoryError(f"unknown colour '{raw_color}'")
                quantity = _parse_quantity(row[cols["quantity"]])
            except InventoryError as e:
                report.errors.append(f"line {line}: {e}")
                continue

            notes = _clean_notes(row[cols["notes"]]) if "notes" in cols else None
            key = (part_num, color_id)
            if key in combined:
                report.duplicate_rows_combined += 1
                combined[key][0] += quantity
                if notes and notes not in (combined[key][1] or ""):
                    combined[key][1] = f"{combined[key][1]}; {notes}" if combined[key][1] else notes
            else:
                combined[key] = [quantity, notes]
            report.rows_imported += 1
            report.pieces_added += quantity

        now = _now()
        conn.executemany(_UPSERT_SQL, [(p, c, q, n, now) for (p, c), (q, n) in combined.items()])

    return report


def export_csv() -> str:
    """Your inventory as CSV text, in the same format import_csv accepts."""
    inv = get_inventory()[["part_num", "color_id", "quantity", "notes", "part_name", "color_name"]]
    return inv.to_csv(index=False)

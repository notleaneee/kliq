"""
projects.py — Reserve pieces for a build, mark it built, take it apart.

A "project" is one copy of a build you're working on. It records, for every
piece of the build, how many are NEEDED and how many it's HOLDING from your
inventory. Held pieces aren't free for other builds:

    available = owned - (pieces held by all active projects)

Life of a project:

    reserve ──▶ reserved ──complete──▶ built ──disassemble──▶ disassembled
                   │                      (pieces released, back in the pool)
                   └──cancel──▶ cancelled (pieces released)

Tracking is optional. Completing a build with keep_pieces=False records it as
built but releases its pieces straight away. Use that if you don't want a
finished model to tie up your inventory.

Released projects keep their rows as history; they just stop counting as held.
Your owned quantities never change here. Disassembling doesn't "add pieces
back": they never left your inventory, they were only held.

With substitutions on, a project may hold a different physical piece than
the one listed (say an older mold). project_parts records what the build
needs; project_holds records the actual pieces taken.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from buildfinder.build_database import get_build, get_build_parts
from buildfinder.compatibility import EXACT, MatchRules, allocate
from buildfinder.database import HELD_PIECES_SQL, connect

RESERVED, BUILT, DISASSEMBLED, CANCELLED = "reserved", "built", "disassembled", "cancelled"
ACTIVE_STATUSES = (RESERVED, BUILT)


class ProjectError(ValueError):
    """Raised for invalid project actions (unknown project, wrong status, not enough pieces)."""


@dataclass
class ProjectResult:
    project_id: int
    pieces_needed: int
    pieces_held: int

    @property
    def pieces_missing(self) -> int:
        return self.pieces_needed - self.pieces_held


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _free_pieces(conn) -> pd.DataFrame:
    """Owned minus held as (part_num, color_id, quantity), inside the caller's transaction."""
    return pd.read_sql_query(
        f"""
        SELECT i.part_num, i.color_id, i.quantity - COALESCE(h.held, 0) AS quantity
        FROM inventory i
        LEFT JOIN ({HELD_PIECES_SQL}) h ON h.part_num = i.part_num AND h.color_id = i.color_id
        WHERE i.quantity - COALESCE(h.held, 0) > 0
        """,
        conn,
    )


_ADD_HOLD_SQL = """
INSERT INTO project_holds (project_id, part_num, color_id, quantity) VALUES (?, ?, ?, ?)
ON CONFLICT (project_id, part_num, color_id) DO UPDATE SET quantity = quantity + excluded.quantity
"""


def _add_holds(conn, project_id: int, holds: pd.DataFrame) -> None:
    conn.executemany(_ADD_HOLD_SQL, [(int(project_id), str(p), int(c), int(q))
                                     for p, c, q in holds.itertuples(index=False, name=None)])


def start_project(conn, build_id: str, name: str, required: pd.DataFrame, status: str,
                  allow_partial: bool = False, notes: Optional[str] = None,
                  rules: MatchRules = EXACT) -> ProjectResult:
    """
    Create a project and hold as many of its pieces as are free.

    Runs inside the caller's connection so it can share one transaction with
    other changes (set import uses that). Most code should call
    reserve_build() or record_built() instead.
    """
    if status not in ACTIVE_STATUSES:
        raise ProjectError(f"a new project must be {RESERVED} or {BUILT}")
    try:
        result = allocate(required, _free_pieces(conn), rules)
    except ValueError:
        raise ProjectError(f"{build_id} has no parts list")

    if result.missing_total and not allow_partial:
        raise ProjectError(
            f"not enough free pieces for {name}: {result.missing_total} of "
            f"{result.required_total} are missing or already held by other projects")

    now = _now()
    cur = conn.execute(
        "INSERT INTO projects (build_id, name, status, pieces_needed, notes, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (build_id, name, status, result.required_total, notes, now, now))
    project_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO project_parts (project_id, part_num, color_id, needed, held) VALUES (?, ?, ?, ?, ?)",
        [(project_id, str(r.part_num), int(r.color_id), int(r.required), int(r.available))
         for r in result.lines.itertuples()])
    _add_holds(conn, project_id, result.holds)
    return ProjectResult(project_id, result.required_total, result.available_total)


def _build_requirements(build_id: str):
    build = get_build(build_id)
    if build is None:
        raise ProjectError(f"unknown build '{build_id}'")
    return build, get_build_parts(build_id)


def reserve_build(build_id: str, allow_partial: bool = False, notes: Optional[str] = None,
                  rules: MatchRules = EXACT) -> ProjectResult:
    """
    Hold the pieces for a build you plan to make.

    By default this fails unless every piece is free. With allow_partial=True
    it holds what's free; top_up() can grab the rest later. `rules` decides
    which substitutes may be used.
    """
    build, required = _build_requirements(build_id)
    with connect() as conn:
        return start_project(conn, build_id, build["name"], required, RESERVED,
                             allow_partial, notes, rules)


def record_built(build_id: str, keep_pieces: bool = True, allow_partial: bool = True,
                 notes: Optional[str] = None, rules: MatchRules = EXACT) -> ProjectResult:
    """Mark a build as built without reserving first (e.g. you built it yesterday)."""
    build, required = _build_requirements(build_id)
    with connect() as conn:
        result = start_project(conn, build_id, build["name"], required, BUILT,
                               allow_partial, notes, rules)
        if not keep_pieces:
            _release(conn, result.project_id)
    return result


def _get(conn, project_id: int) -> dict:
    row = conn.execute("SELECT id, status, released_at FROM projects WHERE id = ?",
                       (int(project_id),)).fetchone()
    if row is None:
        raise ProjectError(f"unknown project {project_id}")
    return {"id": row[0], "status": row[1], "released_at": row[2]}


def _set_status(conn, project_id: int, status: str) -> None:
    conn.execute("UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                 (status, _now(), int(project_id)))


def _release(conn, project_id: int) -> None:
    conn.execute("UPDATE projects SET released_at = ?, updated_at = ? WHERE id = ? AND released_at IS NULL",
                 (_now(), _now(), int(project_id)))


def _require_status(project: dict, allowed: tuple, action: str) -> None:
    if project["status"] not in allowed:
        raise ProjectError(f"can't {action} a project that is {project['status']}")


def complete_project(project_id: int, keep_pieces: bool = True) -> None:
    """reserved -> built. keep_pieces=False releases its pieces (no inventory tracking)."""
    with connect() as conn:
        _require_status(_get(conn, project_id), (RESERVED,), "complete")
        _set_status(conn, project_id, BUILT)
        if not keep_pieces:
            _release(conn, project_id)


def disassemble_project(project_id: int) -> None:
    """built -> disassembled. Its pieces become available again."""
    with connect() as conn:
        _require_status(_get(conn, project_id), (BUILT,), "disassemble")
        _set_status(conn, project_id, DISASSEMBLED)
        _release(conn, project_id)


def cancel_project(project_id: int) -> None:
    """reserved -> cancelled. Its pieces become available again."""
    with connect() as conn:
        _require_status(_get(conn, project_id), (RESERVED,), "cancel")
        _set_status(conn, project_id, CANCELLED)
        _release(conn, project_id)


def top_up(project_id: int, rules: MatchRules = EXACT) -> int:
    """For a project holding fewer pieces than it needs, hold any that are now free. Returns how many."""
    with connect() as conn:
        project = _get(conn, project_id)
        _require_status(project, ACTIVE_STATUSES, "top up")
        if project["released_at"] is not None:
            raise ProjectError("this project isn't tracking pieces")
        short = pd.read_sql_query(
            "SELECT part_num, color_id, needed - held AS quantity FROM project_parts "
            "WHERE project_id = ? AND held < needed",
            conn, params=(int(project_id),))
        if short.empty:
            return 0
        result = allocate(short, _free_pieces(conn), rules)
        extra = result.lines[result.lines["available"] > 0]
        conn.executemany(
            "UPDATE project_parts SET held = held + ? WHERE project_id = ? AND part_num = ? AND color_id = ?",
            [(int(r.available), int(project_id), str(r.part_num), int(r.color_id))
             for r in extra.itertuples()])
        _add_holds(conn, project_id, result.holds)
        if not extra.empty:
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (_now(), int(project_id)))
        return result.available_total


def list_projects() -> pd.DataFrame:
    """All projects, newest first, with how many pieces each is holding right now."""
    with connect() as conn:
        return pd.read_sql_query(
            """
            SELECT p.id, p.build_id, p.name, p.status, p.pieces_needed,
                   CASE WHEN p.released_at IS NULL
                        THEN COALESCE((SELECT SUM(quantity) FROM project_holds ph
                                       WHERE ph.project_id = p.id), 0)
                        ELSE 0 END AS pieces_held,
                   (p.released_at IS NULL) AS tracking,
                   p.notes, p.created_at, p.updated_at
            FROM projects p ORDER BY p.id DESC
            """,
            conn,
        )


def project_holds(project_id: int) -> pd.DataFrame:
    """The physical pieces a project took: part_num, color_id, quantity."""
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT part_num, color_id, quantity FROM project_holds WHERE project_id = ?",
            conn, params=(int(project_id),))


def project_parts(project_id: int) -> pd.DataFrame:
    """A project's lines: part_num, color_id, needed, held, missing."""
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT part_num, color_id, needed, held FROM project_parts WHERE project_id = ?",
            conn, params=(int(project_id),))
    return df.assign(missing=df["needed"] - df["held"])

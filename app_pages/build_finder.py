"""
build_finder.py — The "Build Finder" page: what can I build from the pieces I own?

Part of the combined app. Run from the project root:
    streamlit run app.py

How Streamlit works: every click re-runs this whole script from the top.
Button callbacks (on_click=...) run BEFORE that re-run, so the page always
shows the result of the click. The catalog is cached with @st.cache_data so
it's loaded once, not on every click.
"""

import io
import sqlite3

import pandas as pd
import streamlit as st

from buildfinder import inventory as inv
from buildfinder import piece_usage as pu
from buildfinder import projects as pj
from buildfinder import set_import as si
from buildfinder import substitutions as subs
from buildfinder.build_database import describe_pieces, get_all_build_parts, get_build_parts, get_builds
from buildfinder.compatibility import (ANY_COLOR, STATUS_ALMOST, STATUS_BUILDABLE, STATUS_NEEDS_PARTS,
                               MatchRules, check_build, group_lines, score_builds)
from buildfinder.database import PROJECT_ROOT, catalog_is_loaded, init_db
from buildfinder import recommendations as recs_mod
from buildfinder.recommendations import buildable_share_at_k, rank_builds, status_counts
from swipe import database as ratings_db

PAGE_SIZE = 10
SAMPLE_CSV = PROJECT_ROOT / "samples" / "sample_inventory.csv"
STATUS_TEXT = {STATUS_BUILDABLE: ":green[**Buildable**]",
               STATUS_ALMOST: ":orange[**Almost**]",
               STATUS_NEEDS_PARTS: ":red[**Needs pieces**]"}
STATUS_WORD = {STATUS_BUILDABLE: "Buildable", STATUS_ALMOST: "Almost",
               STATUS_NEEDS_PARTS: "Needs pieces"}
PROJECT_STYLE = {pj.RESERVED: "blue", pj.BUILT: "green", pj.DISASSEMBLED: "gray", pj.CANCELLED: "gray"}

st.set_page_config(page_title="Build Finder · kliq", page_icon=":material/extension:", layout="wide")

init_db()
if not catalog_is_loaded():
    st.error("No catalog yet. Run `python build_finder_catalog.py` first.")
    st.stop()

SWIPE_LABELS = {recs_mod.SWIPE_IGNORE: "Ignore", recs_mod.SWIPE_LIKED_FIRST: "Liked first",
                recs_mod.SWIPE_LIKED_ONLY: "Liked only", recs_mod.SWIPE_HIDE_DISLIKED: "Hide dislikes"}


def load_swipe_ratings():
    """(liked, disliked) set numbers from the Swipe page. Empty if it can't be read."""
    try:
        ratings_db.init_db()
        ratings = ratings_db.get_ratings()
    except sqlite3.Error:
        return set(), set()
    return (set(ratings.loc[ratings["liked"] == 1, "set_num"]),
            set(ratings.loc[ratings["liked"] == 0, "set_num"]))


swipe_liked, swipe_disliked = load_swipe_ratings()

# Defaults for settings the page menu remembers across pages (see app.py).
# They're set here rather than as value=... on the widgets, because a widget
# with its own default whose value is ALSO restored via session_state makes
# Streamlit show a warning.
for _key, _default in {"rule_molds": True, "rec_show": "All", "rec_max_missing": 50,
                       "unlock_max": 10}.items():
    st.session_state.setdefault(_key, _default)

st.session_state.setdefault("flash", None)
st.session_state.setdefault("editor_version", 0)


# ── Cached data ──────────────────────────────────────────────────────────────

@st.cache_data
def load_builds() -> pd.DataFrame:
    return get_builds()


@st.cache_data
def load_build_parts() -> pd.DataFrame:
    return get_all_build_parts()


@st.cache_data
def load_colors() -> pd.DataFrame:
    return inv.all_colors()


COLOR_MODE_LABELS = {"exact": "Exact", "similar": "Old ↔ new greys", "any": "Any"}
RULE_DEFAULTS = {"rule_color": "exact", "rule_molds": True, "rule_alternates": False,
                 "rule_prints": False}


def rule_key() -> tuple:
    """The current matching settings, as a hashable tuple (usable as a cache key)."""
    return tuple(st.session_state.get(k, v) for k, v in RULE_DEFAULTS.items())


@st.cache_resource
def get_rules(color_mode: str, molds: bool, alternates: bool, prints: bool) -> MatchRules:
    return subs.build_rules(color_mode, molds, alternates, prints)


def current_rules() -> MatchRules:
    return get_rules(*rule_key())


def _pieces(pieces_csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(pieces_csv), dtype={"part_num": str})


@st.cache_data(max_entries=20)
def cached_scores(pieces_csv: str, rules_key: tuple) -> pd.DataFrame:
    # Keyed on the pieces' CSV text and the rules: cheap to hash, and any
    # change produces a new key, so scores are recomputed only when needed.
    return score_builds(load_build_parts(), _pieces(pieces_csv), get_rules(*rules_key))


@st.cache_data(max_entries=10)
def cached_group_lines(pieces_csv: str, rules_key: tuple) -> pd.DataFrame:
    return group_lines(load_build_parts(), _pieces(pieces_csv), get_rules(*rules_key))


@st.cache_data(max_entries=10)
def cached_rarely_used(pieces_csv: str, rules_key: tuple, top_ids: tuple) -> pd.DataFrame:
    return pu.rarely_used_pieces(_pieces(pieces_csv), load_build_parts(), top_ids,
                                 get_rules(*rules_key))


@st.cache_data(max_entries=10)
def cached_plan(pieces_csv: str, rules_key: tuple, objective: str) -> pu.Plan:
    pieces = _pieces(pieces_csv)
    return pu.plan_builds(cached_scores(pieces_csv, rules_key), load_build_parts(), pieces,
                          get_rules(*rules_key), objective)


def color_name(color_id) -> str:
    names = dict(zip(load_colors()["id"], load_colors()["name"]))
    return names.get(color_id, str(color_id))


def flash(kind: str, message: str) -> None:
    """Queue a message to show on the next run (survives the rerun)."""
    st.session_state.flash = (kind, message)


def refresh_editors() -> None:
    # Table editors and the file uploader remember their state under their
    # key; bumping the version gives them a clean slate after data changes.
    st.session_state.editor_version += 1


def run_action(action, success_message: str) -> None:
    """Run a project/inventory action, turning expected errors into a message."""
    try:
        result = action()
    except (pj.ProjectError, inv.InventoryError, si.SetImportError) as e:
        flash("error", str(e).capitalize())
        return
    # replace(), not format(): a set name containing "{" would break format().
    flash("success", success_message.replace("{result}", str(result)))
    refresh_editors()


# ── Callbacks ────────────────────────────────────────────────────────────────

def on_reserve(build_id: str, partial: bool) -> None:
    run_action(lambda: pj.reserve_build(build_id, allow_partial=partial, rules=current_rules()),
               "Reserved. See **Projects**.")


def on_reserve_plan(build_ids: list) -> None:
    failed = []
    for build_id in build_ids:
        try:
            pj.reserve_build(build_id, rules=current_rules())
        except pj.ProjectError:
            failed.append(build_id)
    if failed:
        flash("warning", f"Couldn't reserve: {', '.join(failed)}")
    else:
        flash("success", f"Reserved {len(build_ids)} builds. See **Projects**.")
    refresh_editors()


def on_built(build_id: str, keep_key: str) -> None:
    keep = st.session_state[keep_key]
    run_action(lambda: pj.record_built(build_id, keep_pieces=keep, rules=current_rules()),
               "Saved as built.")


def on_complete(project_id: int, keep_key: str) -> None:
    keep = st.session_state[keep_key]
    run_action(lambda: pj.complete_project(project_id, keep_pieces=keep), "Marked as built.")


def on_clear_inventory() -> None:
    inv.clear_inventory()
    st.session_state.confirm_clear = False
    refresh_editors()
    flash("success", "Everything deleted.")


# ── Add-piece pop-up ─────────────────────────────────────────────────────────

PICKER_COLUMNS = 4
PICKER_PAGE = 24


def open_add_piece() -> None:
    st.session_state.show_add_piece = True


def close_add_piece() -> None:
    st.session_state.show_add_piece = False
    st.session_state.pop("add_part", None)
    st.session_state.pop("add_color", None)


def pick_part(part_num: str) -> None:
    st.session_state.add_part = part_num
    st.session_state.add_all_colours = False
    st.session_state.pop("add_color", None)
    st.session_state.pop("add_other_color", None)


def pick_color(color_id: int) -> None:
    st.session_state.add_color = color_id


def use_other_color() -> None:
    if st.session_state.add_other_color is not None:
        st.session_state.add_color = st.session_state.add_other_color


def reset_picker_limit() -> None:
    st.session_state.add_limit = PICKER_PAGE


def change_part() -> None:
    st.session_state.pop("add_part", None)
    st.session_state.pop("add_color", None)
    st.session_state.pop("add_other_color", None)


def show_more_parts() -> None:
    st.session_state.add_limit = st.session_state.get("add_limit", PICKER_PAGE) + PICKER_PAGE


def picture(url, width: int = 120) -> None:
    if isinstance(url, str):
        st.image(url, width=width)
    else:
        st.caption("No picture")


def part_grid() -> None:
    """Step 1: browse pieces by picture."""
    cats = inv.all_categories()
    c1, c2 = st.columns([3, 2])
    query = c1.text_input("Search", placeholder="brick 2 x 4   or   3001", key="add_query",
                          on_change=reset_picker_limit)
    category = c2.selectbox("Type", [None] + list(cats["id"]), key="add_category",
                            on_change=reset_picker_limit,
                            format_func=lambda c: "All types" if c is None
                            else cats.set_index("id").loc[c, "name"])
    limit = st.session_state.get("add_limit", PICKER_PAGE)
    parts = inv.browse_parts(query, category, limit=limit)
    if parts.empty:
        st.caption("No pieces found.")
        return
    for start in range(0, len(parts), PICKER_COLUMNS):
        cols = st.columns(PICKER_COLUMNS)
        for col, row in zip(cols, parts.iloc[start:start + PICKER_COLUMNS].itertuples()):
            with col:
                picture(row.img_url)
                st.caption(f"**{row.part_num}** · {row.name}")
                st.button("Choose", key=f"pick_{row.part_num}", on_click=pick_part,
                          args=(row.part_num,), width="stretch")
    if len(parts) == limit:
        st.button("Show more", on_click=show_more_parts, key="add_more")


def color_grid(part_num: str) -> None:
    """Step 2: pick a colour, seeing the piece in each colour."""
    all_options = inv.part_color_options(part_num)
    chosen = st.session_state.get("add_color")
    if all_options.empty:
        st.caption("No colour pictures for this piece.")
    show_all = st.session_state.get("add_all_colours", False)
    options = all_options if show_all else all_options.head(18)
    for start in range(0, len(options), 6):
        cols = st.columns(6)
        for col, row in zip(cols, options.iloc[start:start + 6].itertuples()):
            with col:
                picture(row.img_url, width=70)
                st.button(row.color_name, key=f"color_{row.color_id}", on_click=pick_color,
                          args=(row.color_id,), width="stretch",
                          type="primary" if row.color_id == chosen else "secondary")
    if len(all_options) > len(options):
        st.button(f"Show all {len(all_options)} colours", key="add_show_colours",
                  on_click=lambda: st.session_state.update(add_all_colours=True))
    with st.expander("Other colour"):
        st.selectbox("Colour", [None] + list(load_colors()["id"]), key="add_other_color",
                     format_func=lambda c: "Choose…" if c is None else color_name(c),
                     on_change=use_other_color)


@st.dialog("Add a piece", width="large", on_dismiss=close_add_piece)
def add_piece_dialog() -> None:
    # A session flag keeps the pop-up open across re-runs, so you can add
    # several pieces in a row. Closing it clears the flag (on_dismiss).
    if st.session_state.pop("add_done", None):
        st.success(st.session_state.pop("add_message"))
        st.session_state.add_qty = 1     # set before the widget is drawn, so it's allowed

    part_num = st.session_state.get("add_part")
    if part_num is None:
        part_grid()
        return

    part = inv.get_part(part_num) or {"part_num": part_num, "name": part_num, "img_url": None}
    left, right = st.columns([1, 3])
    with left:
        picture(part["img_url"], width=110)
    with right:
        st.markdown(f"**{part['name']}**")
        st.caption(f"Part {part_num} · {part.get('category') or ''}")
        st.button("Choose a different piece", key="add_change", on_click=change_part)

    color_grid(part_num)
    colour = st.session_state.get("add_color")
    if colour is None:
        st.caption("Pick a colour.")
        return

    qty = st.number_input("How many", min_value=1, max_value=inv.MAX_QUANTITY, value=1, key="add_qty")
    if st.button(f"Add {qty} in {color_name(colour)}", type="primary", width="stretch",
                 key="add_submit"):
        try:
            inv.add_piece(part_num, colour, qty)
        except inv.InventoryError as e:
            st.error(str(e).capitalize())
        else:
            st.session_state.add_done = True
            st.session_state.add_message = f"Added {qty} × {part['name']} in {color_name(colour)}"
            refresh_editors()
            st.rerun()   # full re-run so the counts update; the flag keeps the pop-up open


# ── Shared data for this run ─────────────────────────────────────────────────

builds = load_builds()
available = inv.get_available_pieces()
totals = inv.inventory_totals()

with st.sidebar:
    st.button("Add piece", type="primary", icon=":material/add:", width="stretch",
              key="open_add_piece", on_click=open_add_piece)

    c1, c2 = st.columns(2)
    c1.metric("Owned", f"{totals['total_pieces']:,}")
    c2.metric("Free", f"{totals['available_pieces']:,}", help="Not used by a project")

    st.divider()
    st.subheader("Matching")
    st.radio("Colour", list(COLOR_MODE_LABELS), format_func=COLOR_MODE_LABELS.get,
             key="rule_color", horizontal=True,
             help="Old ↔ new greys: Light Gray = Light Bluish Gray, Dark Gray = Dark Bluish "
                  "Gray, Brown = Reddish Brown.")
    st.checkbox("Mold variants", key="rule_molds",
                help="Same piece, different factory mold.")
    st.checkbox("Similar parts", key="rule_alternates",
                help="Rebrickable's alternates. Looser, e.g. a magnet brick for a plain brick.")
    st.checkbox("Plain for printed", key="rule_prints",
                help="Build without the print. A print never replaces a different print.")

if st.session_state.get("show_add_piece"):
    add_piece_dialog()

if st.session_state.flash:
    kind, message = st.session_state.flash
    getattr(st, kind)(message)
    st.session_state.flash = None

rules = current_rules()
available_csv = available.to_csv(index=False)
scores = cached_scores(available_csv, rule_key())


# ── Shared pieces of UI ──────────────────────────────────────────────────────

def pieces_table(lines: pd.DataFrame, columns: dict) -> None:
    """Show part lines with names and images. `columns` maps column -> heading."""
    shown = describe_pieces(lines)
    shown.loc[shown["color_id"] == ANY_COLOR, "color_name"] = "any colour"
    config = {"img_url": st.column_config.ImageColumn("", width="small"),
              "part_name": "Piece", "color_name": "Colour", **columns}
    st.dataframe(shown[list(config)], hide_index=True, column_config=config)


def describe_substitutes(subs_list) -> str:
    return ", ".join(f"{q} × {p} {color_name(c)}" for p, c, q in subs_list)


LINE_COLUMNS = {"required": "Need", "owned": "Have", "substitutes_text": "Substitute",
                "missing": "Missing"}


def build_lines_table(lines: pd.DataFrame) -> None:
    pieces_table(lines.assign(substitutes_text=lines["substitutes"].map(describe_substitutes)),
                 LINE_COLUMNS)


def build_card(build: dict, key_prefix: str) -> None:
    build_id = build["build_id"]
    check = check_build(get_build_parts(build_id), available, rules)
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            if isinstance(build.get("img_url"), str):
                st.image(build["img_url"], width="stretch")
        with right:
            st.markdown(f"#### {build['name']}")
            year = int(build["year"]) if pd.notna(build.get("year")) else ""
            st.caption(f"{build['theme']} · {year} · [Instructions]({build['instructions_url']})")
            if build_id in swipe_liked:
                st.markdown(":red[:material/favorite:] Liked in Swipe")
            elif build_id in swipe_disliked:
                st.caption(":material/thumb_down: Disliked in Swipe")
            st.markdown(STATUS_TEXT[check.status])
            st.progress(check.coverage,
                        text=f"{check.available_total} of {check.required_total} pieces"
                             + (f" · {check.missing_total} missing" if check.missing_total else ""))
            if check.substituted_total:
                st.caption(f"Uses {check.substituted_total} substitutes")

            a1, a2 = st.columns(2)
            key = f"{key_prefix}_{build_id}"
            if check.status == STATUS_BUILDABLE:
                a1.button("Reserve", key=f"reserve_{key}", type="primary", width="stretch",
                          on_click=on_reserve, args=(build_id, False),
                          help="Set these pieces aside for this build")
            elif check.available_total > 0:
                a1.button("Reserve what I have", key=f"reserve_{key}", width="stretch",
                          on_click=on_reserve, args=(build_id, True))
            with a2.popover("I built this", width="stretch"):
                st.toggle("Keep pieces reserved", value=True, key=f"keep_{key}")
                st.button("Save", key=f"built_{key}", on_click=on_built,
                          args=(build_id, f"keep_{key}"))

        missing = check.missing_lines
        if not missing.empty:
            with st.expander(f"Missing pieces ({check.missing_total})"):
                build_lines_table(missing)
        with st.expander("All pieces"):
            build_lines_table(check.lines)


# ── Tabs ─────────────────────────────────────────────────────────────────────

projects = pj.list_projects()
active_projects = projects[projects["status"].isin(pj.ACTIVE_STATUSES)]
tab_recs, tab_usage, tab_inv, tab_projects, tab_browse = st.tabs(
    ["What can I build?", "Use my pieces", "My pieces",
     f"Projects ({len(active_projects)})", "Browse"])

with tab_recs:
    if totals["total_pieces"] == 0:
        st.info("No pieces yet. Click **Add piece**, or import a set in **My pieces**.")
    else:
        counts = status_counts(scores)
        c1, c2, c3 = st.columns(3)
        c1.metric("Buildable", counts[STATUS_BUILDABLE])
        c2.metric("Almost", counts[STATUS_ALMOST], help="5 or fewer pieces missing")
        c3.metric("Need more", counts[STATUS_NEEDS_PARTS])

        f1, f2, f3 = st.columns([2, 2, 3])
        show = f1.radio("Show", ["Buildable", "Buildable + almost", "All"], key="rec_show")
        swipe_mode = f1.selectbox(
            "Swipe likes", list(SWIPE_LABELS), format_func=SWIPE_LABELS.get, key="rec_swipe",
            disabled=not (swipe_liked or swipe_disliked),
            help="Use the sets you liked on the Swipe page" if swipe_liked or swipe_disliked
                 else "Rate sets on the Swipe page first")
        max_missing = f2.slider("Max missing (50 = any)", 0, 50, key="rec_max_missing")
        biggest = int(builds["num_parts"].max())
        st.session_state.setdefault("rec_pieces", (1, biggest))
        piece_range = f2.slider("Size (pieces)", 1, biggest, key="rec_pieces")
        search = f3.text_input("Search", key="rec_search", placeholder="name or set number")
        themes = f3.multiselect("Themes", sorted(builds["theme"].dropna().unique()), key="rec_themes")

        statuses = {"Buildable": [STATUS_BUILDABLE],
                    "Buildable + almost": [STATUS_BUILDABLE, STATUS_ALMOST],
                    "All": None}[show]
        ranked = rank_builds(scores, builds, statuses=statuses,
                             max_missing=None if max_missing == 50 else max_missing,
                             min_parts=piece_range[0], max_parts=piece_range[1],
                             search=search, themes=themes)
        ranked = recs_mod.apply_swipe_taste(ranked, swipe_liked, swipe_disliked, swipe_mode)

        share = buildable_share_at_k(rank_builds(scores, builds), 10)
        caption = f"{len(ranked):,} builds · {share:.0%} of the top 10 are buildable"
        if swipe_liked:
            liked_scores = scores[scores["build_id"].isin(swipe_liked)]
            caption += (f" · {len(liked_scores)} of your {len(swipe_liked)} liked sets are here, "
                        f"{int((liked_scores['status'] == STATUS_BUILDABLE).sum())} buildable")
        st.caption(caption)

        if ranked.empty:
            st.write("Nothing matches.")
        else:
            n_pages = (len(ranked) - 1) // PAGE_SIZE + 1
            if st.session_state.get("rec_page", 1) > n_pages:   # filters shrank the list
                st.session_state.rec_page = 1
            page = st.number_input(f"Page (of {n_pages})", 1, n_pages, key="rec_page")
            start = (page - 1) * PAGE_SIZE
            for _, row in ranked.iloc[start:start + PAGE_SIZE].iterrows():
                build_card(row.to_dict(), key_prefix="rec")

with tab_inv:
    # ---- Import a set ----
    st.subheader("Import a set")
    if not si.sets_loaded():
        st.warning("Set data missing. Run `python build_finder_catalog.py` again.")
    else:
        set_query = st.text_input("Set number or name", placeholder="30669  or  Red Plane",
                                  key="set_query")
        found = si.find_sets(set_query) if set_query.strip() else pd.DataFrame()
        if set_query.strip() and found.empty:
            st.caption("No sets found.")
        elif not found.empty:
            labels = {r.set_num: f"{r.name} ({r.set_num}, {r.num_parts} pieces)"
                      for r in found.itertuples()}
            set_num = st.selectbox("Set", list(labels), format_func=labels.get, key="set_pick")
            info = found.set_index("set_num").loc[set_num]
            i1, i2 = st.columns([1, 3])
            if isinstance(info["img_url"], str):
                i1.image(info["img_url"], width="stretch")
            with i2:
                copies = st.number_input("Copies", 1, si.MAX_COPIES, 1, key="set_copies")
                spares = st.checkbox("Include spare pieces", value=True, key="set_spares")
                built = st.checkbox("It's built right now", key="set_built",
                                    help="Its pieces count as in use until you take it apart.")

            contents = si.set_contents(set_num, include_spares=spares)
            contents["quantity"] *= copies
            with st.expander("Lost some pieces?"):
                editor_rows = describe_pieces(contents).assign(missing=0)
                edited = st.data_editor(
                    editor_rows[["img_url", "part_name", "color_name", "quantity", "missing",
                                 "part_num", "color_id"]],
                    # Rows change with copies/spares, so those are part of the key:
                    # otherwise edits would be re-applied to the wrong rows.
                    key=f"missing_{set_num}_{copies}_{spares}_{st.session_state.editor_version}",
                    hide_index=True,
                    disabled=["img_url", "part_name", "color_name", "quantity"],
                    column_order=["img_url", "part_name", "color_name", "quantity", "missing"],
                    column_config={
                        "img_url": st.column_config.ImageColumn("", width="small"),
                        "part_name": "Piece", "color_name": "Colour", "quantity": "In set",
                        "missing": st.column_config.NumberColumn("Lost", min_value=0, step=1),
                    },
                )
            if st.button("Import", type="primary", key="import_set"):
                lost = edited[edited["missing"].fillna(0) > 0][["part_num", "color_id", "missing"]]
                lost = lost.rename(columns={"missing": "quantity"})
                try:
                    report = si.import_set(set_num, copies=int(copies), include_spares=spares,
                                           missing=lost, currently_built=built)
                except si.SetImportError as e:
                    flash("error", str(e).capitalize())
                else:
                    flash("success", f"Added {report.pieces_added:,} pieces from {report.name}.")
                    refresh_editors()
                st.rerun()

    # ---- CSV import ----
    st.divider()
    st.subheader("Import a list")
    st.caption("CSV with columns `part_num`, `color_id`, `quantity`")
    up1, up2 = st.columns([3, 1])
    # Versioned key: after an import the uploader is reset, so the same file
    # can't be imported twice by accident.
    uploaded = up1.file_uploader("CSV file", type=["csv"], label_visibility="collapsed",
                                 key=f"uploader_{st.session_state.editor_version}")
    if SAMPLE_CSV.exists():
        up2.download_button("Example file", SAMPLE_CSV.read_bytes(),
                            file_name="sample_inventory.csv", mime="text/csv")
    if uploaded is not None and st.button("Import file"):
        try:
            report = inv.import_csv(uploaded.getvalue())
        except inv.InventoryError as e:
            flash("error", str(e).capitalize())
        else:
            message = f"Added {report.pieces_added:,} pieces."
            if report.errors:
                shown = "\n".join(f"- {e}" for e in report.errors[:10])
                more = f"\n- ...and {len(report.errors) - 10} more" if len(report.errors) > 10 else ""
                flash("warning", f"{message} Skipped {len(report.errors)} rows:\n{shown}{more}")
            else:
                flash("success", message)
            refresh_editors()
        st.rerun()

    # ---- The inventory table ----
    st.divider()
    st.subheader("My pieces")
    table = inv.get_inventory()
    if table.empty:
        st.write("No pieces yet.")
    else:
        filt = st.text_input("Filter", placeholder="Filter", key="inv_filter",
                             label_visibility="collapsed")
        if filt.strip():
            needle = filt.strip().lower()
            mask = table[["part_num", "part_name", "color_name", "notes", "category"]].fillna("") \
                .apply(lambda col: col.str.lower().str.contains(needle, regex=False)).any(axis=1)
            table = table[mask]
        table = table.assign(remove=False).reset_index(drop=True)

        edited = st.data_editor(
            table[["img_url", "part_name", "color_name", "quantity", "held", "notes", "remove"]],
            key=f"inv_editor_{st.session_state.editor_version}",
            hide_index=True,
            disabled=["img_url", "part_name", "color_name", "held"],
            column_config={
                "img_url": st.column_config.ImageColumn("", width="small"),
                "part_name": "Piece", "color_name": "Colour",
                "quantity": st.column_config.NumberColumn("Owned", min_value=0,
                                                          max_value=inv.MAX_QUANTITY, step=1),
                "held": st.column_config.NumberColumn("In use"),
                "notes": st.column_config.TextColumn("Notes"),
                "remove": st.column_config.CheckboxColumn("Remove"),
            },
        )

        s1, s2, s3 = st.columns([1, 1, 2])
        if s1.button("Save changes", type="primary"):
            try:
                changed = inv.apply_table_edits(table, edited)
            except inv.InventoryError as e:
                flash("error", str(e).capitalize())
            else:
                flash("success", "Saved." if changed else "Nothing changed.")
            refresh_editors()
            st.rerun()
        s2.download_button("Export", inv.export_csv(), file_name="my_lego_pieces.csv",
                           mime="text/csv")
        with s3:
            if st.checkbox("Delete everything?", key="confirm_clear"):
                st.button("Yes, delete", on_click=on_clear_inventory)

with tab_projects:
    if projects.empty:
        st.write("No projects yet. Reserve a build to start one.")
    for p in active_projects.itertuples():
        with st.container(border=True):
            info = si.get_set(p.build_id) or {}
            left, right = st.columns([1, 3])
            if isinstance(info.get("img_url"), str):
                left.image(info["img_url"], width="stretch")
            with right:
                st.markdown(f"#### {p.name}")
                st.markdown(f":{PROJECT_STYLE[p.status]}[**{p.status.capitalize()}**]")
                if p.tracking:
                    st.progress(p.pieces_held / p.pieces_needed,
                                text=f"{p.pieces_held} of {p.pieces_needed} pieces set aside")

                b1, b2, b3 = st.columns(3)
                if p.status == pj.RESERVED:
                    with b1.popover("Mark built", width="stretch"):
                        st.toggle("Keep pieces reserved", value=True, key=f"pkeep_{p.id}")
                        st.button("Confirm", key=f"complete_{p.id}", on_click=on_complete,
                                  args=(p.id, f"pkeep_{p.id}"))
                    b2.button("Cancel", key=f"cancel_{p.id}", width="stretch",
                              on_click=run_action,
                              args=(lambda pid=p.id: pj.cancel_project(pid), "Cancelled."))
                else:
                    b1.button("Take apart", key=f"disassemble_{p.id}", width="stretch",
                              help="Frees its pieces", on_click=run_action,
                              args=(lambda pid=p.id: pj.disassemble_project(pid), "Taken apart."))
                if p.tracking and p.pieces_held < p.pieces_needed:
                    b3.button("Grab new pieces", key=f"topup_{p.id}", width="stretch",
                              on_click=run_action,
                              args=(lambda pid=p.id: pj.top_up(pid, rules=current_rules()),
                                    "Added {result} pieces."))

            lines = pj.project_parts(p.id)
            short = lines[lines["missing"] > 0]
            if p.tracking and not short.empty:
                with st.expander(f"Missing ({int(short['missing'].sum())})"):
                    pieces_table(short, {"needed": "Need", "missing": "Missing"})
            if p.tracking:
                with st.expander("Pieces set aside"):
                    pieces_table(pj.project_holds(p.id), {"quantity": "How many"})

    history = projects[~projects["status"].isin(pj.ACTIVE_STATUSES)]
    if not history.empty:
        st.subheader("History")
        st.dataframe(history[["name", "status", "updated_at"]], hide_index=True,
                     column_config={"name": "Build", "status": "Status", "updated_at": "Date"})

with tab_usage:
    if available.empty:
        st.info("No free pieces yet.")
    else:
        name_of = dict(zip(builds["build_id"], builds["name"]))
        free_total = int(available["quantity"].sum())

        st.subheader("Build several at once")
        st.caption("Each piece is used only once.")
        objective = st.radio("Aim for", [pu.PLAN_MOST_PIECES, pu.PLAN_MOST_BUILDS], horizontal=True,
                             key="plan_objective",
                             format_func={pu.PLAN_MOST_PIECES: "Most pieces used",
                                          pu.PLAN_MOST_BUILDS: "Most builds"}.get)
        plan = cached_plan(available_csv, rule_key(), objective)
        if plan.builds.empty:
            st.write("Nothing is buildable yet.")
        else:
            st.metric("Pieces used", f"{plan.pieces_used:,} of {free_total:,}")
            st.dataframe(plan.builds.assign(name=plan.builds["build_id"].map(name_of))
                         [["name", "pieces"]],
                         hide_index=True, column_config={"name": "Build", "pieces": "Pieces"})
            st.button(f"Reserve all {len(plan.builds)}", on_click=on_reserve_plan,
                      args=(list(plan.builds["build_id"]),), key="reserve_plan")

        st.divider()
        st.subheader("Biggest builds for your pieces")
        share = pu.inventory_share(rank_builds(scores, builds), free_total)
        if share.empty:
            st.write("Nothing is buildable or almost buildable yet.")
        else:
            st.dataframe(share.assign(status=share["status"].map(STATUS_WORD))
                         [["name", "status", "share"]],
                         hide_index=True,
                         column_config={"name": "Build", "status": "Status",
                                        "share": st.column_config.ProgressColumn(
                                            "Uses of your pieces", format="percent",
                                            min_value=0, max_value=1)})

        st.divider()
        st.subheader("Pieces worth getting")
        max_extra = st.slider("For builds missing up to", 1, 20, key="unlock_max")
        unlock = pu.unlockable_pieces(cached_group_lines(available_csv, rule_key()), scores,
                                      max_missing=max_extra)
        if unlock.empty:
            st.write("No builds are that close yet.")
        else:
            shown = unlock.head(25).rename(columns={"part_key": "part_num", "color_key": "color_id"})
            pieces_table(shown, {"pieces_needed": "Get", "builds_unlocked": "Completes",
                                 "builds_helped": "Helps"})

        st.divider()
        st.subheader("Least-used pieces")
        top_ids = tuple(rank_builds(scores, builds).head(50)["build_id"])
        rare = cached_rarely_used(available_csv, rule_key(), top_ids)
        pieces_table(rare.head(25), {"quantity": "You have", "builds_using": "Builds using it"})

with tab_browse:
    bq = st.text_input("Search", key="browse_search", placeholder="name, set number or theme")
    shown = builds
    if bq.strip():
        needle = bq.strip().lower()
        shown = builds[builds[["name", "build_id", "theme"]].apply(
            lambda col: col.astype(str).str.lower().str.contains(needle, regex=False)).any(axis=1)]
    shown = shown.reset_index(drop=True)
    st.caption("Click a row to see it.")
    selection = st.dataframe(
        shown[["name", "theme", "year", "num_parts"]],
        hide_index=True, on_select="rerun", selection_mode="single-row", key="browse_table",
        column_config={"name": "Name", "theme": "Theme",
                       "year": st.column_config.NumberColumn("Year", format="%d"),
                       "num_parts": "Pieces"},
    )
    rows = selection.selection.rows
    if rows:
        build_card(shown.iloc[rows[0]].to_dict(), key_prefix="browse")

"""
swipe.py — The "Swipe" page: rate LEGO sets, get personal recommendations.

Part of the combined app. Run from the project root:
    streamlit run app.py

How Streamlit works (the one thing to understand):
    Every time you click a button, Streamlit re-runs this ENTIRE script from
    top to bottom. Normal variables are reset on every run. Anything that must
    survive between runs goes in `st.session_state` (in memory, per browser tab)
    or in SQLite (on disk, survives restarts). Ratings go to SQLite; the
    "which set am I looking at right now" pointer goes in session_state.
"""

import sqlite3
from functools import wraps
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from swipe import database as db
from swipe import evaluation as ev
from swipe import recommender as rec
from swipe import similarity as sim
from swipe.catalog import load_catalog
from swipe.embeddings import load_embeddings
from swipe.card import swipe_card

TOP_K = 5
METHOD_MODEL, METHOD_SIMILAR = "model", "similar"
METHOD_LABELS = {METHOD_MODEL: "Learned from you", METHOD_SIMILAR: "Looks like your likes"}

st.set_page_config(page_title="Swipe · kliq", page_icon=":material/toys:", layout="centered")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def get_catalog() -> pd.DataFrame:
    # @st.cache_data runs this once and reuses the result on every rerun,
    # instead of re-reading the CSV on every button click.
    return load_catalog()


@st.cache_data
def get_embeddings():
    return load_embeddings()


@st.cache_resource
def get_faiss_index():
    # cache_resource (not cache_data): a FAISS index is a live object that
    # can't be copied/pickled, so one shared instance is kept.
    return sim.build_index(get_embeddings()[0])


@st.cache_data(max_entries=50)
def get_recommendations(ratings_key: tuple, method: str):
    # Keyed on the ratings themselves, so the model retrains exactly when
    # they change (each new rating = a new key).
    ratings = pd.DataFrame(list(ratings_key), columns=["set_num", "liked"])
    embeddings, index_df = get_embeddings()
    if method == METHOD_MODEL:
        return rec.recommend(catalog, ratings, embeddings, index_df, k=TOP_K)
    return sim.baseline_recommend(catalog, ratings, get_faiss_index(), embeddings, index_df, k=TOP_K)


@st.cache_data(max_entries=20)
def get_evaluation(ratings_key: tuple):
    ratings = pd.DataFrame(list(ratings_key), columns=["set_num", "liked"])
    embeddings, index_df = get_embeddings()
    return ev.evaluate(ratings, embeddings, index_df), ev.compare_C(ratings, embeddings, index_df)


@st.cache_data(max_entries=200)
def get_similar(set_num: str):
    embeddings, index_df = get_embeddings()
    return sim.similar_sets(set_num, get_faiss_index(), embeddings, index_df, catalog, k=4)


try:
    catalog = get_catalog()
    get_embeddings()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

if catalog.empty:
    st.error("The catalog is empty. Rebuild it with:  python build_swipe_catalog.py")
    st.stop()

try:
    db.init_db()
    ratings = db.get_ratings()
except sqlite3.Error as e:
    st.error(f"Can't open the ratings database ({db.get_db_path()}): {e}")
    st.stop()

# session_state keys survive reruns. setdefault only sets them on the first run.
st.session_state.setdefault("current_set", None)
st.session_state.setdefault("skipped", set())


# ── Picking the next set ─────────────────────────────────────────────────────

def pick_next_set(rated: set) -> Optional[str]:
    """Random unrated, unskipped set; None once everything has been rated."""
    unrated = catalog.loc[~catalog["set_num"].isin(rated), "set_num"]
    if unrated.empty:
        return None
    candidates = unrated[~unrated.isin(st.session_state.skipped)]
    if candidates.empty:
        # Everything left was skipped; show the skipped sets again.
        st.session_state.skipped = set()
        candidates = unrated
    return candidates.sample(1).iloc[0]


def show_image(path) -> None:
    # The catalog is cached, so an image deleted while the app runs would
    # otherwise crash st.image.
    if isinstance(path, str) and Path(path).exists():
        st.image(path, width="stretch")
    else:
        st.caption("Image missing")


# ── Button callbacks ─────────────────────────────────────────────────────────
# on_click callbacks run BEFORE the script reruns, so the rerun already sees
# the updated database. That avoids the classic "click twice" Streamlit bug.

def db_safe(callback):
    """Show database errors (e.g. "database is locked") as a message, not a crash."""
    @wraps(callback)
    def wrapper(*args, **kwargs):
        try:
            callback(*args, **kwargs)
        except sqlite3.Error as e:
            st.session_state.db_error = f"Couldn't save: {e}"
    return wrapper


@db_safe
def on_rate(liked: bool) -> None:
    set_num = st.session_state.current_set
    if set_num is None:
        return
    db.save_rating(set_num, liked)
    st.session_state.current_set = None   # forces a new pick on the rerun


@db_safe
def on_swipe() -> None:
    value = st.session_state.get(f"swipe_{st.session_state.current_set}")
    # Ignore a swipe that arrives for a card we've already moved past.
    if value and value.get("set_num") == st.session_state.current_set:
        on_rate(bool(value["liked"]))


@db_safe
def on_rate_set(set_num: str, liked: bool) -> None:
    db.save_rating(set_num, liked)


def on_skip() -> None:
    if st.session_state.current_set is not None:
        st.session_state.skipped.add(st.session_state.current_set)
    st.session_state.current_set = None


@db_safe
def on_undo() -> None:
    last = db.get_last_rating()
    if last is None:
        return
    db.delete_rating(last["set_num"])
    st.session_state.current_set = last["set_num"]   # show it again


@db_safe
def on_reset() -> None:
    db.clear_ratings()
    st.session_state.current_set = None
    st.session_state.skipped = set()
    st.session_state.confirm_reset = False


# ── Sidebar: progress + controls ─────────────────────────────────────────────

if "db_error" in st.session_state:
    st.error(st.session_state.pop("db_error"))

rated_set_nums = set(ratings["set_num"])
counts = db.rating_counts()
readiness = rec.check_readiness(ratings)

with st.sidebar:
    c1, c2 = st.columns(2)
    c1.metric("Likes", counts["likes"])
    c2.metric("Dislikes", counts["dislikes"])
    st.caption(f"{counts['total']} of {len(catalog):,} rated")
    if not readiness.ready:
        st.progress(min(counts["total"] / rec.MIN_RATINGS, 1.0))
        st.caption(readiness.message)

    st.divider()
    st.button("Undo", on_click=on_undo, disabled=counts["total"] == 0, width="stretch")
    if st.checkbox("Delete all ratings?", key="confirm_reset"):
        st.button("Yes, delete", type="primary", on_click=on_reset, width="stretch")


# ── Main area ────────────────────────────────────────────────────────────────

tab_rate, tab_for_you, tab_history, tab_test = st.tabs(["Rate", "For you", "My ratings", "Test"])
ratings_key = tuple(ratings[["set_num", "liked"]].itertuples(index=False, name=None))

with tab_rate:
    current = st.session_state.current_set
    # Pick a new set if we don't have one, if it was rated (e.g. in another
    # browser tab), or if it's no longer in the catalog.
    if (current is None or current in rated_set_nums
            or current not in set(catalog["set_num"])):
        current = pick_next_set(rated_set_nums)
        st.session_state.current_set = current

    if current is None:
        st.success("You've rated every set.")
        st.caption("Add more with `python build_swipe_catalog.py --n 2000`.")
    else:
        row = catalog.loc[catalog["set_num"] == current].iloc[0]

        st.subheader(row["name"])
        if Path(row["local_image_path"]).exists():
            swipe_card(current, row["local_image_path"], alt=row["name"],
                       key=f"swipe_{current}", on_change=on_swipe)
        else:
            st.caption("Image missing")

        m1, m2, m3 = st.columns(3)
        m1.metric("Year", int(row["year"]))
        m2.metric("Pieces", f"{int(row['num_parts']):,}")
        m3.metric("Set", row["set_num"])
        st.caption(f"{row['theme_name']} · "
                   f"[Rebrickable](https://rebrickable.com/sets/{row['set_num']}/)")

        b1, b2, b3 = st.columns([2, 1, 2])
        b1.button("Dislike", on_click=on_rate, args=(False,), width="stretch",
                  icon=":material/thumb_down:", key="dislike")
        b2.button("Skip", on_click=on_skip, width="stretch", key="skip")
        b3.button("Like", on_click=on_rate, args=(True,), type="primary", width="stretch",
                  icon=":material/thumb_up:", key="like")

        with st.expander("Looks similar"):
            similar = get_similar(current)
            cols = st.columns(4)
            for col, item in zip(cols, similar.itertuples()):
                with col:
                    show_image(item.local_image_path)
                    st.caption(f"{item.name} · {item.similarity:.2f}")

with tab_for_you:
    method = st.radio("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get,
                      horizontal=True, key="rec_method",
                      help="Learned from you: a model trained on your likes AND dislikes. "
                           "Looks like your likes: the sets closest to the average of your likes.")
    if method == METHOD_MODEL and not readiness.ready:
        st.info(f"The model unlocks soon. {readiness.message} "
                "Until then, try **Looks like your likes**.")
    elif method == METHOD_SIMILAR and counts["likes"] == 0:
        st.info("Like at least one set first.")
    else:
        recs = get_recommendations(ratings_key, method)
        if recs is None or recs.empty:
            st.write("No unrated sets left.")
        else:
            for r in recs.itertuples():
                with st.container(border=True):
                    left, right = st.columns([1, 2])
                    with left:
                        show_image(r.local_image_path)
                    with right:
                        st.markdown(f"#### {r.name}")
                        st.caption(f"{r.theme_name} · {int(r.year)} · {int(r.num_parts):,} pieces")
                        if method == METHOD_MODEL:
                            st.progress(float(r.score), text=f"{r.score:.0%} match")
                        else:
                            st.progress(float(min(max(r.similarity, 0), 1)),
                                        text=f"Similarity {r.similarity:.2f}")
                        b1, b2 = st.columns(2)
                        b1.button("Dislike", key=f"rec_dislike_{r.set_num}", width="stretch",
                                  icon=":material/thumb_down:", on_click=on_rate_set,
                                  args=(r.set_num, False))
                        b2.button("Like", key=f"rec_like_{r.set_num}", width="stretch",
                                  type="primary", icon=":material/thumb_up:",
                                  on_click=on_rate_set, args=(r.set_num, True))

with tab_history:
    if ratings.empty:
        st.write("No ratings yet.")
    else:
        history = ratings.merge(catalog, on="set_num", how="left").iloc[::-1]
        liked_tab, disliked_tab = st.tabs([f"Liked ({counts['likes']})",
                                           f"Disliked ({counts['dislikes']})"])
        for container, value in ((liked_tab, 1), (disliked_tab, 0)):
            subset = history[history["liked"] == value]
            with container:
                if subset.empty:
                    st.write("None yet.")
                    continue
                cols = st.columns(4)
                for i, (_, r) in enumerate(subset.iterrows()):
                    with cols[i % 4]:
                        show_image(r["local_image_path"])
                        st.caption(f"{r['name']} ({int(r['year'])})"
                                   if pd.notna(r["year"]) else r["set_num"])

with tab_test:
    st.markdown("**If we pick 5 sets, how many would you like?**")
    st.caption(f"We hide {ev.TEST_FRACTION:.0%} of your ratings, recommend from them, "
               f"and check your real answers. Repeated {ev.N_SPLITS}×.")
    missing = ev.check_eval_readiness(ratings)
    if missing:
        st.info(missing)
    else:
        result, by_C = get_evaluation(ratings_key)
        k = result.k
        model, baseline = result.mean(ev.MODEL), result.mean(ev.BASELINE)
        random_pick = result.mean(ev.RANDOM)
        t1, t2, t3 = st.columns(3)
        t1.metric(METHOD_LABELS[METHOD_MODEL], f"{model * k:.1f} / {k}",
                  delta=f"{(model - random_pick) * k:+.1f} vs random")
        t2.metric(METHOD_LABELS[METHOD_SIMILAR], f"{baseline * k:.1f} / {k}",
                  delta=f"{(baseline - random_pick) * k:+.1f} vs random")
        t3.metric("Random", f"{random_pick * k:.1f} / {k}")

        wins, ties, losses = result.head_to_head()
        if model > baseline:
            st.success(f"The model wins: better in {wins} of {result.n_splits} tests "
                       f"({ties} ties, {losses} worse).")
        elif model < baseline:
            st.warning(f"The simple method wins: the model is worse in {losses} "
                       f"of {result.n_splits} tests.")
        else:
            st.info("It's a tie.")
        st.caption(f"Precision@{k}: model {model:.2f}, similarity {baseline:.2f}, "
                   f"random {random_pick:.2f}, best possible {result.mean(ev.BEST):.2f}. "
                   f"{result.n_ratings} ratings, {result.test_size} hidden per test.")

        with st.expander("Model setting (C)"):
            st.caption("Higher C = trusts your ratings more. Just for exploring: "
                       f"the score above always uses C = {rec.DEFAULT_C:g}.")
            st.dataframe(by_C.assign(**{f"liked of {k}": by_C["precision"] * k})
                         .drop(columns="precision"),
                         hide_index=True, column_config={
                             "C": st.column_config.NumberColumn(format="%g"),
                             f"liked of {k}": st.column_config.NumberColumn(format="%.1f")})

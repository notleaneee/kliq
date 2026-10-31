"""
app.py — kliq: one app, two pages:

    Swipe          rate LEGO sets, get recommendations      (app_pages/swipe.py)
    Build Finder   what can I build from my pieces?          (app_pages/build_finder.py)

Run from the project root:
    streamlit run app.py

st.navigation draws the page menu at the top of the sidebar and runs the
chosen page's script. Each page is still a normal Streamlit script.
"""

from pathlib import Path

import streamlit as st

from buildfinder.database import unpack_if_needed

ASSETS = Path(__file__).resolve().parent / "assets"

unpack_if_needed()          # hosted copy: data/lego.db.gz -> data/lego.db (first start only)

# The app name, top-left of the sidebar.
st.logo(str(ASSETS / "kliq_logo.svg"), size="large")

PAGES = {
    "swipe": st.Page("app_pages/swipe.py", title="Swipe", icon=":material/swipe:",
                     url_path="swipe", default=True),
    "build_finder": st.Page("app_pages/build_finder.py", title="Build Finder",
                            icon=":material/extension:", url_path="build_finder"),
}

# Settings worth remembering while you're on the other page. Streamlit
# forgets a widget's value as soon as you switch to a page that doesn't draw
# it. So we keep a copy under a separate name ("_kept_<key>") that Streamlit
# never cleans up.
#
# The copy must be put back in the SAME run that draws the page: only a
# value set during that run is sent to the browser. (Restoring it earlier
# leaves the server thinking "Any" while the browser still shows "Exact".)
# Pages set their defaults with setdefault, not value=..., so restoring
# doesn't trigger Streamlit's "default value + Session State" warning.
KEEP = {
    "swipe": ["rec_method"],
    "build_finder": ["rule_color", "rule_molds", "rule_alternates", "rule_prints",
                     "rec_show", "rec_max_missing", "rec_pieces", "rec_search", "rec_themes",
                     "rec_swipe", "plan_objective", "unlock_max", "inv_filter"],
}

page = st.navigation(list(PAGES.values()))
current = next(name for name, p in PAGES.items() if p is page)
just_arrived = st.session_state.get("_last_page") != current
st.session_state["_last_page"] = current

for name, keys in KEEP.items():
    for key in keys:
        kept = f"_kept_{key}"
        if name == current and just_arrived:
            if kept in st.session_state:
                st.session_state[key] = st.session_state[kept]   # redraw with the saved value
        elif key in st.session_state:
            st.session_state[kept] = st.session_state[key]       # remember the latest value

page.run()

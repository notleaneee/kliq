"""
swipe.py — Python side of the swipe card (components/swipe_card/index.html).

    value = swipe_card(set_num, image_path, key=..., on_change=...)

returns None until you swipe, then {"set_num": "...", "liked": True/False}.
The image is sent inline as a data URI (base64 text), because the card's
iframe can't read files from your disk.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Callable, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent.parent / "components" / "swipe_card"
_swipe_card = components.declare_component("swipe_card", path=str(_COMPONENT_DIR))


def image_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def swipe_card(set_num: str, image_path: str, alt: str = "", key: Optional[str] = None,
               on_change: Optional[Callable[[], None]] = None):
    return _swipe_card(set_num=set_num, image=image_data_uri(image_path), alt=alt,
                       key=key, on_change=on_change, default=None)

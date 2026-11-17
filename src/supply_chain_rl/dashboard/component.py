"""The ``network3d`` Streamlit component (D-43): a three.js scene in its own frame.

The frame stays in place from one day to the next (same key), so the camera is kept and the
browser moves towers and shipments smoothly to the new day. All values come from
``scene3d.scene_spec``. The files are in ``network3d/``; three.js r170 is saved under
``network3d/vendor/`` (MIT licence), so no network access is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

FRONTEND = Path(__file__).parent / "network3d"
_network3d = components.declare_component("network3d", path=str(FRONTEND))


def network3d(spec: dict[str, Any], key: str) -> None:
    """Draw (or move to) the scene of ``spec``."""
    _network3d(spec=spec, key=key, default=None)

from __future__ import annotations

import os


def apply():
    """ Find the right Qt """
    try:
        import PyQt6
    except ImportError:
        return

    plugins = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

    if os.path.isdir(plugins):
        os.environ.setdefault("QT_PLUGIN_PATH", plugins)
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                              os.path.join(plugins, "platforms"))


apply()

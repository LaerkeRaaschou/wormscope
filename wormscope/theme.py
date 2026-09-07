from functools import lru_cache
from importlib.resources import files
from string import Template

from . import config

# Collect colors from config file, ready for the stylesheet
PALETTE = {
    "bg":          config.BG_LIGHT,
    "panel":       config.PANEL,
    "grid":        config.GRIDCOL,
    "blue":        config.DARK_BLUE,
    "blue_hi":     config.DARK_BLUE_HI,
    "blue_lo":     config.DARK_BLUE_LO,
    "blue_edge":   "#a9c2e6",
    "disabled_bg": "#9fb0c8",
    "disabled_fg": "#e8eef7",
}


@lru_cache(maxsize=1) # Cache stylesheet
def stylesheet() -> str:
    """ Load style.qss and substitute the palette """
    raw = files(__package__).joinpath("resources/style.qss").read_text(encoding="utf-8")
    return Template(raw).substitute(PALETTE)


def apply_theme(app) -> None:
    """Call once, right after constructing QApplication."""
    app.setStyleSheet(stylesheet())

from __future__ import annotations

import os
import sys
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

project = "fo-mo-cid"
author = "fo-mo-cid contributors"
copyright = "2026, fo-mo-cid contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

root_doc = "index"
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]

html_theme = "furo"
html_title = "fo-mo-cid"
html_static_path: list[str] = []
templates_path = ["_templates"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = "off"
nb_execution_timeout = -1
nb_render_markdown_format = "myst"

nitpicky = False
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_inherit_docstrings = False

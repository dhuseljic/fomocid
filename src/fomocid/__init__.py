"""Core package for tutorial-first SSL workflows."""

from pathlib import Path

__all__ = ["__version__", "DATA_ROOT"]

__version__ = "0.1.0"

# One level above the fomocid repo root: .../Code/fomocid/src/fomocid/ → × 4
DATA_ROOT: Path = Path(__file__).parent.parent.parent.parent

"""Filesystem and reproducibility helpers shared by scripts and tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch RNGs via Lightning helpers.

    Args:
        seed: Random seed value.

    Returns:
        The seed value returned by Lightning after initialization.

    Parameters
    ----------
    seed : int
        Input value for ``seed``.

    Returns
    -------
    result : int
        Return value produced by the function.
    """
    return pl.seed_everything(seed, workers=True)


def create_run_dir(output_dir: str | Path, stem: str) -> Path:
    """Create a timestamped run directory.

    Args:
        output_dir: Parent directory for experiment outputs.
        stem: Prefix used before the timestamp in the run directory name.

    Returns:
        Newly created run directory path.

    Parameters
    ----------
    output_dir : str | Path
        Input value for ``output_dir``.
    stem : str
        Input value for ``stem``.

    Returns
    -------
    result : Path
        Return value produced by the function.
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_yaml(output_path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialize a mapping to YAML.

    Args:
        output_path: Destination path for the YAML file.
        payload: Mapping to serialize.

    Returns:
        Path to the saved file.

    Parameters
    ----------
    output_path : str | Path
        Input value for ``output_path``.
    payload : dict[str, Any]
        Input value for ``payload``.

    Returns
    -------
    result : Path
        Return value produced by the function.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path


def save_json(output_path: str | Path, payload: dict[str, Any]) -> Path:
    """Serialize a mapping to JSON.

    Args:
        output_path: Destination path for the JSON file.
        payload: Mapping to serialize.

    Returns:
        Path to the saved file.

    Parameters
    ----------
    output_path : str | Path
        Input value for ``output_path``.
    payload : dict[str, Any]
        Input value for ``payload``.

    Returns
    -------
    result : Path
        Return value produced by the function.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path

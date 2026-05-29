"""YAML config loading and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from fomocid.config_types import RootConfig, normalize_root_config


def load_config(config_path: str | Path) -> RootConfig:
    """Load, normalize, and validate a YAML tutorial config.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Normalized config with concrete values for the ``seed``, ``dataset``,
        ``ssl``, ``optimizer``, ``trainer``, ``evaluation``, and ``analysis``
        sections.

    Raises:
        ValueError: If the YAML document does not deserialize to a mapping.

    Parameters
    ----------
    config_path : str | Path
        Input value for ``config_path``.

    Returns
    -------
    result : RootConfig
        Return value produced by the function.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must deserialize to a mapping.")
    return normalize_root_config(config)

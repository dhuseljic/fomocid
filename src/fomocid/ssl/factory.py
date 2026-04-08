"""Factories for instantiating SSL tutorial modules from config."""

from __future__ import annotations

from typing import Any

from fomocid.config_types import RootConfig

from .dino import DINOModule
from .mae import MAEModule


METHOD_REGISTRY = {
    "mae": MAEModule,
    "dino": DINOModule,
}


def get_ssl_method_name(config: RootConfig) -> str:
    """Resolve the configured SSL method name.

    Args:
        config: Normalized root config whose ``ssl.method`` selects the SSL
            implementation.

    Returns:
        The configured method name.

    Raises:
        ValueError: If the method is not supported.
    """
    method_name = config["ssl"].get("method")
    if method_name not in METHOD_REGISTRY:
        supported = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unsupported SSL method: {method_name}. Supported methods: {supported}")
    return str(method_name)


def build_ssl_module(config: RootConfig) -> Any:
    """Instantiate the configured SSL Lightning module.

    Args:
        config: Normalized root config used to choose and construct the SSL
            module.

    Returns:
        Configured Lightning module for the selected SSL method.
    """
    method_name = get_ssl_method_name(config)
    return METHOD_REGISTRY[method_name](config)


def load_ssl_module(checkpoint_path: str, config: RootConfig) -> Any:
    """Load a checkpointed SSL Lightning module for the configured method.

    Args:
        checkpoint_path: Path to the saved Lightning checkpoint.
        config: Normalized root config whose ``ssl.method`` determines which
            module class should deserialize the checkpoint.

    Returns:
        Loaded Lightning module ready for evaluation or feature extraction.
    """
    method_name = get_ssl_method_name(config)
    module_cls = METHOD_REGISTRY[method_name]
    return module_cls.load_from_checkpoint(checkpoint_path, config=config)

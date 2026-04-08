"""Self-supervised learning modules."""

from .dino import DINOModule
from .factory import build_ssl_module, get_ssl_method_name, load_ssl_module
from .mae import MAEModule

__all__ = [
    "DINOModule",
    "MAEModule",
    "build_ssl_module",
    "get_ssl_method_name",
    "load_ssl_module",
]

"""Feature extraction and downstream evaluation utilities."""

from .features import (
    evaluate_embeddings,
    extract_features,
    fit_linear_probe,
    knn_accuracy,
    save_embedding_bundle,
)

__all__ = [
    "evaluate_embeddings",
    "extract_features",
    "fit_linear_probe",
    "knn_accuracy",
    "save_embedding_bundle",
]


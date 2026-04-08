"""Feature extraction and downstream embedding evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fomocid.config_types import KNNConfig, LinearProbeConfig


@torch.no_grad()
def extract_features(
    model: Any,
    dataloader: Any,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    """Embed a labeled dataset with a trained SSL model.

    Args:
        model: Trained model that exposes an ``embed`` method returning feature
            vectors for a batch of images.
        dataloader: Iterable yielding ``(inputs, targets)`` batches.
        device: Optional compute device. Defaults to CUDA when available and
            falls back to CPU otherwise.
        max_batches: Optional cap on the number of dataloader batches to
            process. Useful for smoke tests or quick notebook previews.

    Returns:
        Dictionary containing concatenated ``embeddings``, ``labels``, and
        ``images`` tensors on CPU memory.
    """
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(resolved_device)
    model.eval()

    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    images: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = batch
        inputs = inputs.to(resolved_device)
        batch_embeddings = model.embed(inputs).cpu()
        embeddings.append(batch_embeddings)
        labels.append(targets.cpu())
        images.append(inputs.cpu())

    return {
        "embeddings": torch.cat(embeddings, dim=0),
        "labels": torch.cat(labels, dim=0),
        "images": torch.cat(images, dim=0),
    }


def fit_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Train a linear classifier on frozen embeddings.

    Args:
        train_embeddings: Feature matrix for the probe training split with
            shape ``(num_samples, embedding_dim)``.
        train_labels: Integer class labels aligned with ``train_embeddings``.
        test_embeddings: Feature matrix for evaluation.
        test_labels: Integer class labels aligned with ``test_embeddings``.
        num_classes: Number of target classes.
        epochs: Number of linear-probe optimization epochs.
        lr: Probe learning rate.
        weight_decay: Weight decay applied to the probe optimizer.
        batch_size: Mini-batch size for probe training.
        device: Optional compute device for the probe optimization.

    Returns:
        Dictionary containing the best observed test accuracy and the trained
        probe ``state_dict`` moved back to CPU tensors.
    """
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier = nn.Linear(train_embeddings.shape[1], num_classes).to(resolved_device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_embeddings = train_embeddings.to(resolved_device)
    train_labels = train_labels.to(resolved_device)
    test_embeddings = test_embeddings.to(resolved_device)
    test_labels = test_labels.to(resolved_device)

    best_test_accuracy = 0.0
    for _epoch in range(epochs):
        permutation = torch.randperm(train_embeddings.size(0), device=resolved_device)
        for start in range(0, train_embeddings.size(0), batch_size):
            indices = permutation[start : start + batch_size]
            batch_embeddings = train_embeddings[indices]
            batch_labels = train_labels[indices]
            logits = classifier(batch_embeddings)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = classifier(test_embeddings)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == test_labels).float().mean().item()
            best_test_accuracy = max(best_test_accuracy, accuracy)

    return {
        "linear_probe_accuracy": best_test_accuracy,
        "state_dict": {key: value.detach().cpu() for key, value in classifier.state_dict().items()},
    }


@torch.no_grad()
def knn_accuracy(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 5,
    chunk_size: int = 512,
) -> tuple[float, torch.Tensor]:
    """Evaluate embeddings with cosine-similarity k-nearest neighbors.

    Args:
        train_embeddings: Reference embedding bank used for neighbor search.
        train_labels: Labels for the reference embedding bank.
        test_embeddings: Query embeddings to classify.
        test_labels: Ground-truth labels for the queries.
        k: Number of nearest neighbors used for majority voting.
        chunk_size: Number of query embeddings processed at once to limit peak
            memory use.

    Returns:
        Tuple of ``(accuracy, predicted_labels)`` for the test embeddings.
    """
    normalized_train = F.normalize(train_embeddings, dim=1)
    normalized_test = F.normalize(test_embeddings, dim=1)

    predictions: list[torch.Tensor] = []
    for start in range(0, normalized_test.size(0), chunk_size):
        batch = normalized_test[start : start + chunk_size]
        similarities = batch @ normalized_train.T
        topk_indices = similarities.topk(k=min(k, similarities.size(1)), dim=1).indices
        topk_labels = train_labels[topk_indices]
        batch_predictions = torch.mode(topk_labels, dim=1).values
        predictions.append(batch_predictions.cpu())

    predicted_labels = torch.cat(predictions, dim=0)
    accuracy = (predicted_labels == test_labels.cpu()).float().mean().item()
    return accuracy, predicted_labels


@torch.no_grad()
def topk_nearest_neighbors(
    reference_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k cosine-similarity neighbors for a query embedding set.

    Args:
        reference_embeddings: Embeddings searched as the retrieval bank.
        query_embeddings: Embeddings whose nearest neighbors should be found.
        k: Number of neighbors to return per query.

    Returns:
        Tuple of ``(indices, scores)`` on CPU, where indices point into the
        reference embedding tensor.
    """
    normalized_reference = F.normalize(reference_embeddings, dim=1)
    normalized_query = F.normalize(query_embeddings, dim=1)
    similarities = normalized_query @ normalized_reference.T
    scores, indices = similarities.topk(k=min(k, similarities.size(1)), dim=1)
    return indices.cpu(), scores.cpu()


def save_embedding_bundle(
    output_path: str | Path,
    train_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
) -> Path:
    """Persist train/test embedding bundles to disk.

    Args:
        output_path: Destination ``.pt`` file.
        train_features: Feature bundle returned by :func:`extract_features` for
            the training split.
        test_features: Feature bundle returned by :func:`extract_features` for
            the evaluation split.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": train_features, "test": test_features}, output_path)
    return output_path


def evaluate_embeddings(
    train_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    num_classes: int,
    linear_probe_config: LinearProbeConfig,
    knn_config: KNNConfig,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run the downstream metrics used throughout the tutorials.

    Args:
        train_features: Feature bundle for the labeled training split.
        test_features: Feature bundle for the held-out evaluation split.
        num_classes: Number of classes for the linear probe.
        linear_probe_config: Normalized ``evaluation.linear_probe`` settings,
            including ``epochs``, ``lr``, ``weight_decay``, and ``batch_size``.
        knn_config: Normalized ``evaluation.knn`` settings, including ``k``
            and ``chunk_size``.
        device: Optional compute device for the linear probe.

    Returns:
        JSON-serializable summary of the linear-probe and kNN results.
    """
    probe = fit_linear_probe(
        train_features["embeddings"],
        train_features["labels"],
        test_features["embeddings"],
        test_features["labels"],
        num_classes=num_classes,
        epochs=int(linear_probe_config.get("epochs", 25)),
        lr=float(linear_probe_config.get("lr", 1e-2)),
        weight_decay=float(linear_probe_config.get("weight_decay", 0.0)),
        batch_size=int(linear_probe_config.get("batch_size", 256)),
        device=device,
    )
    knn_acc, predicted_labels = knn_accuracy(
        train_features["embeddings"],
        train_features["labels"],
        test_features["embeddings"],
        test_features["labels"],
        k=int(knn_config.get("k", 5)),
        chunk_size=int(knn_config.get("chunk_size", 512)),
    )
    metrics = {
        "linear_probe_accuracy": probe["linear_probe_accuracy"],
        "knn_accuracy": knn_acc,
        "num_train_embeddings": int(train_features["embeddings"].shape[0]),
        "num_test_embeddings": int(test_features["embeddings"].shape[0]),
    }
    metrics["knn_predictions_shape"] = list(predicted_labels.shape)
    return metrics


def save_metrics(output_path: str | Path, metrics: dict[str, Any]) -> Path:
    """Write evaluation metrics to JSON.

    Args:
        output_path: Destination JSON path.
        metrics: Metrics dictionary to serialize.

    Returns:
        Path to the saved metrics file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return output_path

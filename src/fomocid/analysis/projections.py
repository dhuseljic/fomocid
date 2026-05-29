"""Projection and retrieval helpers used by evaluation tutorials."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


def project_embeddings(
    embeddings: torch.Tensor,
    method: str = "pca",
    n_components: int = 2,
) -> tuple[torch.Tensor, str]:
    """Project embeddings to a low-dimensional space for visualization.

    Args:
        embeddings: Feature matrix with shape ``(num_samples, embedding_dim)``.
        method: Projection method name. Supported values are ``"pca"`` and
            ``"tsne"``.
        n_components: Number of output projection dimensions.

    Returns:
        Tuple of ``(projected_embeddings, resolved_method)``.

    Parameters
    ----------
    embeddings : torch.Tensor
        Input value for ``embeddings``.
    method : str
        Input value for ``method``.
    n_components : int
        Input value for ``n_components``.

    Returns
    -------
    result : tuple[torch.Tensor, str]
        Return value produced by the function.
    """
    normalized_method = method.strip().lower().replace("-", "")
    if normalized_method == "pca":
        return _project_embeddings_pca(embeddings, n_components=n_components), "pca"
    if normalized_method == "tsne":
        return _project_embeddings_tsne(embeddings, n_components=n_components), "tsne"

    warnings.warn(f"Projection method '{method}' is not available; falling back to PCA.")
    return _project_embeddings_pca(embeddings, n_components=n_components), "pca"


def _project_embeddings_pca(embeddings: torch.Tensor, *, n_components: int) -> torch.Tensor:
    """Project embeddings with PCA.

    Parameters
    ----------
    embeddings : torch.Tensor
        Input value for ``embeddings``.
    n_components : int
        Input value for ``n_components``.

    Returns
    -------
    result : torch.Tensor
        Return value produced by the function.
    """
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _u, _s, v = torch.pca_lowrank(centered, q=max(n_components, 2))
    projection = centered @ v[:, :n_components]
    return projection.cpu()


def _project_embeddings_tsne(embeddings: torch.Tensor, *, n_components: int) -> torch.Tensor:
    """Project embeddings with t-SNE.

    Parameters
    ----------
    embeddings : torch.Tensor
        Input value for ``embeddings``.
    n_components : int
        Input value for ``n_components``.

    Returns
    -------
    result : torch.Tensor
        Return value produced by the function.
    """
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "t-SNE projection requires scikit-learn. Install the project dependencies again after pulling this change."
        ) from exc

    num_samples = int(embeddings.shape[0])
    if num_samples < 3:
        warnings.warn("t-SNE requires at least 3 samples; falling back to PCA.")
        return _project_embeddings_pca(embeddings, n_components=n_components)

    perplexity = min(30.0, max(2.0, float(num_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=n_components,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=0,
    )
    projected = tsne.fit_transform(embeddings.detach().cpu().float().numpy())
    return torch.from_numpy(np.asarray(projected)).cpu()


def _color_palette() -> list[tuple[int, int, int]]:
    """Return a small categorical color palette for class visualizations.

    Parameters
    ----------
    None
        This function takes no explicit input parameters.

    Returns
    -------
    result : list[tuple[int, int, int]]
        Return value produced by the function.
    """
    return [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]


def save_projection_figure(
    projected_embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | Path,
    title: str = "Embedding Projection",
    size: tuple[int, int] = (960, 720),
) -> Path:
    """Render a simple scatter plot of projected embeddings as a PNG image.

    Args:
        projected_embeddings: Two-dimensional embedding coordinates.
        labels: Integer labels used to color the points.
        output_path: Destination image path.
        title: Figure title shown at the top-left of the canvas.
        size: Output figure size in pixels as ``(width, height)``.

    Returns:
        Path to the saved figure.

    Parameters
    ----------
    projected_embeddings : torch.Tensor
        Input value for ``projected_embeddings``.
    labels : torch.Tensor
        Input value for ``labels``.
    output_path : str | Path
        Input value for ``output_path``.
    title : str
        Input value for ``title``.
    size : tuple[int, int]
        Input value for ``size``.

    Returns
    -------
    result : Path
        Return value produced by the function.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = size
    margin = 60
    canvas = Image.new("RGB", size, color=(248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill=(15, 23, 42))

    coords = projected_embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    x_values = coords[:, 0]
    y_values = coords[:, 1]

    def scale(values: np.ndarray, lower: int, upper: int) -> np.ndarray:
        """Run the scale operation.

        Parameters
        ----------
        values : np.ndarray
            Input value for ``values``.
        lower : int
            Input value for ``lower``.
        upper : int
            Input value for ``upper``.

        Returns
        -------
        result : np.ndarray
            Return value produced by the function.
        """
        min_value = float(values.min())
        max_value = float(values.max())
        if math.isclose(min_value, max_value):
            return np.full_like(values, (lower + upper) / 2.0)
        normalized = (values - min_value) / (max_value - min_value)
        return normalized * (upper - lower) + lower

    xs = scale(x_values, margin, width - margin)
    ys = scale(-y_values, margin, height - margin)
    palette = _color_palette()

    draw.rectangle((margin, margin, width - margin, height - margin), outline=(148, 163, 184), width=2)
    for x_coord, y_coord, label in zip(xs, ys, labels_np, strict=False):
        color = palette[int(label) % len(palette)]
        draw.ellipse((x_coord - 3, y_coord - 3, x_coord + 3, y_coord + 3), fill=color, outline=color)

    canvas.save(output_path)
    return output_path


def retrieve_nearest_neighbors(
    reference_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest neighbors via cosine similarity.

    Args:
        reference_embeddings: Embeddings searched as the retrieval bank.
        query_embeddings: Embeddings used as search queries.
        k: Number of neighbors to return per query.

    Returns:
        Tuple of ``(indices, scores)`` on CPU.

    Parameters
    ----------
    reference_embeddings : torch.Tensor
        Input value for ``reference_embeddings``.
    query_embeddings : torch.Tensor
        Input value for ``query_embeddings``.
    k : int
        Input value for ``k``.

    Returns
    -------
    result : tuple[torch.Tensor, torch.Tensor]
        Return value produced by the function.
    """
    normalized_reference = F.normalize(reference_embeddings, dim=1)
    normalized_query = F.normalize(query_embeddings, dim=1)
    similarities = normalized_query @ normalized_reference.T
    scores, indices = similarities.topk(k=min(k, similarities.size(1)), dim=1)
    return indices.cpu(), scores.cpu()


def _unnormalize_image(image_tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    """Map a normalized image tensor back to the display range ``[0, 1]``.

    Args:
        image_tensor: Normalized image tensor with shape ``(3, H, W)``.
        mean: Channel-wise normalization mean.
        std: Channel-wise normalization standard deviation.

    Returns:
        Image tensor clamped to display range on CPU.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Input value for ``image_tensor``.
    mean : Iterable[float]
        Input value for ``mean``.
    std : Iterable[float]
        Input value for ``std``.

    Returns
    -------
    result : torch.Tensor
        Return value produced by the function.
    """
    mean_tensor = torch.tensor(list(mean)).view(3, 1, 1)
    std_tensor = torch.tensor(list(std)).view(3, 1, 1)
    image = image_tensor.cpu() * std_tensor + mean_tensor
    return image.clamp(0.0, 1.0)


def build_nearest_neighbor_figure(
    reference_images: torch.Tensor,
    query_images: torch.Tensor,
    neighbor_indices: torch.Tensor,
    output_path: str | Path,
    mean: Iterable[float],
    std: Iterable[float],
    query_labels: torch.Tensor | None = None,
    reference_labels: torch.Tensor | None = None,
    label_names: Iterable[str] | None = None,
    cell_size: int = 128,
) -> Path:
    """Create a query-versus-neighbors image grid for qualitative retrieval.

    Args:
        reference_images: Images corresponding to the retrieval bank.
        query_images: Images that act as retrieval queries.
        neighbor_indices: Neighbor index tensor with shape
            ``(num_queries, num_neighbors)``.
        output_path: Destination image path.
        mean: Channel-wise normalization mean used during preprocessing.
        std: Channel-wise normalization standard deviation used during
            preprocessing.
        query_labels: Optional labels shown next to query images.
        reference_labels: Optional labels shown above retrieved neighbors.
        label_names: Optional mapping from integer labels to display names.
        cell_size: Size of each image cell in pixels.

    Returns:
        Path to the saved figure.

    Parameters
    ----------
    reference_images : torch.Tensor
        Input value for ``reference_images``.
    query_images : torch.Tensor
        Input value for ``query_images``.
    neighbor_indices : torch.Tensor
        Input value for ``neighbor_indices``.
    output_path : str | Path
        Input value for ``output_path``.
    mean : Iterable[float]
        Input value for ``mean``.
    std : Iterable[float]
        Input value for ``std``.
    query_labels : torch.Tensor | None
        Input value for ``query_labels``.
    reference_labels : torch.Tensor | None
        Input value for ``reference_labels``.
    label_names : Iterable[str] | None
        Input value for ``label_names``.
    cell_size : int
        Input value for ``cell_size``.

    Returns
    -------
    result : Path
        Return value produced by the function.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_queries = query_images.size(0)
    num_neighbors = neighbor_indices.size(1)
    header_height = 28
    image = Image.new(
        "RGB",
        ((num_neighbors + 1) * cell_size, num_queries * (cell_size + header_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(image)
    resolved_label_names = list(label_names) if label_names is not None else None

    def format_label(prefix: str, label: int) -> str:
        """Run the format label operation.

        Parameters
        ----------
        prefix : str
            Input value for ``prefix``.
        label : int
            Input value for ``label``.

        Returns
        -------
        result : str
            Return value produced by the function.
        """
        if resolved_label_names is None or label < 0 or label >= len(resolved_label_names):
            return f"{prefix}={label}"
        return f"{prefix}={resolved_label_names[label]}"

    for row in range(num_queries):
        row_top = row * (cell_size + header_height)
        query_label = "Query"
        if query_labels is not None:
            query_label = format_label("Query", int(query_labels[row]))
        draw.text((8, row_top + 6), query_label, fill=(15, 23, 42))

        query_pil = to_pil_image(_unnormalize_image(query_images[row], mean, std))
        image.paste(query_pil.resize((cell_size, cell_size)), (0, row_top + header_height))

        for col, ref_index in enumerate(neighbor_indices[row].tolist(), start=1):
            reference_pil = to_pil_image(_unnormalize_image(reference_images[ref_index], mean, std))
            image.paste(reference_pil.resize((cell_size, cell_size)), (col * cell_size, row_top + header_height))
            if reference_labels is not None:
                draw.text(
                    (col * cell_size + 8, row_top + 6),
                    format_label("nn", int(reference_labels[ref_index])),
                    fill=(15, 23, 42),
                )

    image.save(output_path)
    return output_path

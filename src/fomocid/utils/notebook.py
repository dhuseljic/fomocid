"""Notebook-facing helpers for previews, summaries, and artifact display."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.transforms.functional import to_pil_image

from fomocid.config_types import RootConfig
from fomocid.data import create_datamodule
from fomocid.utils.config import load_config


CLASS_NAMES: dict[str, list[str]] = {
    "cifar10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "mnist": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ],
    "stl10": [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ],
}


def _ensure_config(config_or_path: RootConfig | str | Path) -> RootConfig:
    """Normalize notebook inputs to an in-memory root config.

    Parameters
    ----------
    config_or_path : RootConfig | str | Path
        Input value for ``config_or_path``.

    Returns
    -------
    result : RootConfig
        Return value produced by the function.
    """
    if isinstance(config_or_path, dict):
        return config_or_path
    return load_config(config_or_path)


def _load_raw_dataset(config: RootConfig, split: str) -> tuple[Any, Any]:
    """Load an unaugmented dataset split for notebook visualizations.

    Parameters
    ----------
    config : RootConfig
        Input value for ``config``.
    split : str
        Input value for ``split``.

    Returns
    -------
    result : tuple[Any, Any]
        Return value produced by the function.
    """
    datamodule = create_datamodule(config)
    if datamodule.use_fake_data:
        dataset = FakeData(
            size=datamodule.fake_size,
            image_size=(3, datamodule.image_size, datamodule.image_size),
            num_classes=datamodule.num_classes,
            transform=transforms.Resize((datamodule.image_size, datamodule.image_size)),
        )
        return datamodule, dataset

    dataset = datamodule._build_base_dataset(
        split,
        transform=transforms.Resize((datamodule.image_size, datamodule.image_size)),
        download=datamodule.download,
    )
    return datamodule, dataset


def _dataset_targets(dataset: Any) -> list[int] | None:
    """Extract integer targets from common torchvision dataset attributes.

    Parameters
    ----------
    dataset : Any
        Input value for ``dataset``.

    Returns
    -------
    result : list[int] | None
        Return value produced by the function.
    """
    if hasattr(dataset, "targets"):
        return [int(value) for value in list(dataset.targets)]
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        return [int(value) for value in list(labels)]
    return None


def _label_name(dataset_name: str, label: int) -> str:
    """Map an integer class label to a human-readable class name.

    Parameters
    ----------
    dataset_name : str
        Input value for ``dataset_name``.
    label : int
        Input value for ``label``.

    Returns
    -------
    result : str
        Return value produced by the function.
    """
    names = CLASS_NAMES.get(dataset_name)
    if names is None or label < 0 or label >= len(names):
        return str(label)
    return names[label]


def _to_pil_image(
    image: Image.Image | torch.Tensor,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
) -> Image.Image:
    """Convert tensors or PIL images into a display-ready RGB PIL image.

    Parameters
    ----------
    image : Image.Image | torch.Tensor
        Input value for ``image``.
    mean : Iterable[float] | None
        Input value for ``mean``.
    std : Iterable[float] | None
        Input value for ``std``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    tensor = image.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(list(mean)).view(3, 1, 1)
        std_tensor = torch.tensor(list(std)).view(3, 1, 1)
        tensor = tensor * std_tensor + mean_tensor
    tensor = tensor.clamp(0.0, 1.0)
    return to_pil_image(tensor)


def _fit_image(image: Image.Image, cell_size: int) -> Image.Image:
    """Resize an image to fit inside a square canvas while preserving aspect.

    Parameters
    ----------
    image : Image.Image
        Input value for ``image``.
    cell_size : int
        Input value for ``cell_size``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    image = image.convert("RGB")
    background = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
    resized = image.copy()
    resized.thumbnail((cell_size, cell_size))
    offset = ((cell_size - resized.width) // 2, (cell_size - resized.height) // 2)
    background.paste(resized, offset)
    return background


def _resize_preview_image(image: Image.Image | torch.Tensor, size: int, *, mean: Iterable[float] | None = None, std: Iterable[float] | None = None) -> Image.Image:
    """Convert an image to PIL and upscale it for notebook-friendly previews.

    Parameters
    ----------
    image : Image.Image | torch.Tensor
        Input value for ``image``.
    size : int
        Input value for ``size``.
    mean : Iterable[float] | None
        Input value for ``mean``.
    std : Iterable[float] | None
        Input value for ``std``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    pil_image = _to_pil_image(image, mean=mean, std=std)
    return pil_image.resize((size, size), resample=Image.Resampling.NEAREST)


def build_image_grid(
    images: Sequence[Image.Image | torch.Tensor],
    labels: Sequence[str] | None = None,
    title: str | None = None,
    cell_size: int = 128,
    columns: int = 4,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
) -> Image.Image:
    """Compose a list of images into a labeled grid figure.

    Args:
        images: Images to display.
        labels: Optional captions aligned with ``images``.
        title: Optional title rendered above the grid.
        cell_size: Size of each image cell in pixels.
        columns: Number of columns in the grid.
        mean: Optional normalization mean used to unnormalize tensor images.
        std: Optional normalization standard deviation used to unnormalize
            tensor images.

    Returns:
        PIL image containing the rendered grid.

    Raises:
        ValueError: If ``images`` is empty.

    Parameters
    ----------
    images : Sequence[Image.Image | torch.Tensor]
        Input value for ``images``.
    labels : Sequence[str] | None
        Input value for ``labels``.
    title : str | None
        Input value for ``title``.
    cell_size : int
        Input value for ``cell_size``.
    columns : int
        Input value for ``columns``.
    mean : Iterable[float] | None
        Input value for ``mean``.
    std : Iterable[float] | None
        Input value for ``std``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    if not images:
        raise ValueError("build_image_grid requires at least one image.")

    prepared_images = [_fit_image(_to_pil_image(image, mean=mean, std=std), cell_size=cell_size) for image in images]
    labels = list(labels or [""] * len(prepared_images))
    label_height = 24
    title_height = 30 if title else 0
    padding = 12
    rows = math.ceil(len(prepared_images) / columns)
    width = columns * cell_size + (columns + 1) * padding
    height = rows * (cell_size + label_height) + (rows + 1) * padding + title_height
    canvas = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((padding, 8), title, fill=(15, 23, 42))

    for index, image in enumerate(prepared_images):
        row = index // columns
        col = index % columns
        x = padding + col * (cell_size + padding)
        y = padding + title_height + row * (cell_size + label_height + padding)
        canvas.paste(image, (x, y))
        if index < len(labels) and labels[index]:
            draw.text((x, y + cell_size + 4), labels[index], fill=(51, 65, 85))

    return canvas


def load_config_text(config_or_path: RootConfig | str | Path) -> str:
    """Return a normalized configuration mapping as YAML-formatted text.

    Args:
        config_or_path: Normalized config mapping or path to a YAML config.

    Returns:
        YAML text with the normalized section structure used by the tutorials.

    Parameters
    ----------
    config_or_path : RootConfig | str | Path
        Input value for ``config_or_path``.

    Returns
    -------
    result : str
        Return value produced by the function.
    """
    config = _ensure_config(config_or_path)
    return yaml.safe_dump(config, sort_keys=False)


def build_dataset_preview(
    config_or_path: RootConfig | str | Path,
    split: str,
    limit: int = 12,
    samples_per_class: int = 1,
    seed: int = 7,
    display_size: int | None = None,
) -> Image.Image:
    """Build a labeled preview grid for a dataset split.

    Args:
        config_or_path: Normalized tutorial config or path to one on disk.
        split: Dataset split name to preview.
        limit: Maximum number of examples to display.
        samples_per_class: Number of examples to sample per class when labels
            are available.
        seed: Random seed used for fallback sampling.
        display_size: Render size in pixels for each preview image before
            composing the grid. When omitted, uses a notebook-friendly default.

    Returns:
        PIL image containing a dataset preview grid.

    Parameters
    ----------
    config_or_path : RootConfig | str | Path
        Input value for ``config_or_path``.
    split : str
        Input value for ``split``.
    limit : int
        Input value for ``limit``.
    samples_per_class : int
        Input value for ``samples_per_class``.
    seed : int
        Input value for ``seed``.
    display_size : int | None
        Input value for ``display_size``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    config = _ensure_config(config_or_path)
    datamodule, dataset = _load_raw_dataset(config, split=split)
    dataset_name = config["dataset"]["name"]
    targets = _dataset_targets(dataset)
    preview_size = int(display_size or max(192, datamodule.image_size * 6))

    if targets and len(set(targets)) > 1 and samples_per_class > 0:
        grouped: dict[int, list[int]] = defaultdict(list)
        for idx, target in enumerate(targets):
            grouped[int(target)].append(idx)
        indices: list[int] = []
        for label in sorted(grouped):
            indices.extend(grouped[label][:samples_per_class])
        indices = indices[:limit]
    else:
        rng = np.random.default_rng(seed)
        indices = list(range(min(limit, len(dataset))))
        if len(dataset) > len(indices):
            indices = rng.choice(len(dataset), size=min(limit, len(dataset)), replace=False).tolist()

    images: list[Image.Image] = []
    labels: list[str] = []
    for idx in indices:
        image, label = dataset[idx]
        images.append(_resize_preview_image(image, preview_size))
        labels.append(_label_name(dataset_name, int(label)))

    return build_image_grid(
        images,
        labels=labels,
        title=f"{dataset_name.upper()} {split} split preview",
        cell_size=preview_size,
        columns=4,
    )


def build_pretrain_view_preview(
    config_or_path: RootConfig | str | Path,
    split: str,
    index: int = 0,
    repeats: int = 4,
    display_size: int | None = None,
) -> Image.Image:
    """Visualize the augmented views produced for SSL pretraining.

    Args:
        config_or_path: Normalized tutorial config or path to one on disk.
        split: Dataset split name to sample from.
        index: Dataset example index to visualize.
        repeats: Number of MAE views to sample. Ignored for DINO, where the
            configured multi-crop transform determines the number of views.
        display_size: Render size in pixels for each preview image before
            composing the grid.

    Returns:
        PIL image showing the raw image and the SSL views.

    Raises:
        ValueError: If the configured SSL method is unsupported.

    Parameters
    ----------
    config_or_path : RootConfig | str | Path
        Input value for ``config_or_path``.
    split : str
        Input value for ``split``.
    index : int
        Input value for ``index``.
    repeats : int
        Input value for ``repeats``.
    display_size : int | None
        Input value for ``display_size``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    config = _ensure_config(config_or_path)
    datamodule, dataset = _load_raw_dataset(config, split=split)
    image, _label = dataset[index]
    method_name = config["ssl"]["method"]
    preview_size = int(display_size or max(192, datamodule.image_size * 6))
    images: list[Image.Image] = [_resize_preview_image(image, preview_size)]
    labels: list[str] = ["raw image"]

    if method_name == "mae":
        for view_idx in range(repeats):
            view = datamodule.ssl_transform(image)[0]
            images.append(_resize_preview_image(view, preview_size, mean=datamodule.mean, std=datamodule.std))
            labels.append(f"MAE view {view_idx + 1}")
    elif method_name == "dino":
        views = datamodule.ssl_transform(image)
        images.extend(
            _resize_preview_image(view, preview_size, mean=datamodule.mean, std=datamodule.std) for view in views
        )
        labels.extend(["global 1", "global 2"])
        labels.extend([f"local {idx + 1}" for idx in range(len(views) - 2)])
    else:
        raise ValueError(f"Unsupported method for preview: {method_name}")

    return build_image_grid(
        images,
        labels=labels,
        title=f"{method_name.upper()} pretraining views",
        cell_size=preview_size,
        columns=3 if method_name == "dino" else 4,
    )


def build_mae_mask_preview(
    config_or_path: RootConfig | str | Path,
    split: str = "train",
    index: int = 0,
    seed: int = 7,
    display_size: int | None = None,
) -> Image.Image:
    """Render a toy visualization of MAE patch masking.

    Args:
        config_or_path: Normalized tutorial config or path to one on disk.
        split: Dataset split name to sample from.
        index: Dataset example index to visualize.
        seed: Random seed used to choose the masked patches.
        display_size: Render size in pixels for each preview image before
            composing the grid.

    Returns:
        PIL image showing the resized image, patch grid, and masked version.

    Raises:
        ValueError: If the configuration is not for an MAE tutorial.

    Parameters
    ----------
    config_or_path : RootConfig | str | Path
        Input value for ``config_or_path``.
    split : str
        Input value for ``split``.
    index : int
        Input value for ``index``.
    seed : int
        Input value for ``seed``.
    display_size : int | None
        Input value for ``display_size``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    config = _ensure_config(config_or_path)
    if config["ssl"]["method"] != "mae":
        raise ValueError("build_mae_mask_preview is only valid for MAE configs.")

    datamodule, dataset = _load_raw_dataset(config, split=split)
    base_image, _label = dataset[index]
    preview_size = int(display_size or max(224, datamodule.image_size * 8))
    base = _resize_preview_image(base_image, preview_size)
    patch_size = int(config["ssl"]["patch_size"])
    mask_ratio = float(config["ssl"]["mask_ratio"])
    patches_per_side = datamodule.image_size // patch_size
    display_patch_size = preview_size // patches_per_side
    num_patches = patches_per_side * patches_per_side
    num_masked = int(num_patches * mask_ratio)

    rng = np.random.default_rng(seed)
    masked_indices = set(rng.choice(num_patches, size=num_masked, replace=False).tolist())

    grid = base.copy()
    grid_draw = ImageDraw.Draw(grid)
    for position in range(0, preview_size + 1, display_patch_size):
        grid_draw.line((position, 0, position, preview_size), fill=(100, 116, 139), width=1)
        grid_draw.line((0, position, preview_size, position), fill=(100, 116, 139), width=1)

    masked = base.copy().convert("RGBA")
    overlay = Image.new("RGBA", masked.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for patch_idx in masked_indices:
        row = patch_idx // patches_per_side
        col = patch_idx % patches_per_side
        x0 = col * display_patch_size
        y0 = row * display_patch_size
        x1 = x0 + display_patch_size
        y1 = y0 + display_patch_size
        overlay_draw.rectangle((x0, y0, x1, y1), fill=(15, 23, 42, 160), outline=(226, 232, 240, 180))

    masked = Image.alpha_composite(masked, overlay).convert("RGB")
    return build_image_grid(
        [base, grid, masked],
        labels=[
            "resized image",
            f"{patches_per_side}x{patches_per_side} patch grid",
            f"{num_masked}/{num_patches} masked patches",
        ],
        title="MAE masking intuition",
        cell_size=preview_size,
        columns=3,
    )


def summarize_ssl_batch_shapes(config_or_path: RootConfig | str | Path) -> dict[str, Any]:
    """Inspect the tensor shapes produced by the SSL training dataloader.

    Args:
        config_or_path: Normalized tutorial config or path to one on disk.

    Returns:
        Small summary dictionary describing the number and shapes of the views
        produced by the dataloader.

    Parameters
    ----------
    config_or_path : RootConfig | str | Path
        Input value for ``config_or_path``.

    Returns
    -------
    result : dict[str, Any]
        Return value produced by the function.
    """
    config = _ensure_config(config_or_path)
    datamodule = create_datamodule(config)
    datamodule.setup()
    views, targets = next(iter(datamodule.ssl_train_dataloader()))
    if config["ssl"]["method"] == "mae":
        return {
            "num_views": len(views),
            "view_1_shape": tuple(views[0].shape),
            "targets_shape": tuple(targets.shape),
        }
    return {
        "num_views": len(views),
        "view_shapes": [tuple(view.shape) for view in views],
        "targets_shape": tuple(targets.shape),
    }


def open_saved_image(path: str | Path) -> Image.Image:
    """Open an image artifact from disk as RGB.

    Args:
        path: Path to the image file on disk.

    Returns:
        Loaded RGB PIL image.

    Parameters
    ----------
    path : str | Path
        Input value for ``path``.

    Returns
    -------
    result : Image.Image
        Return value produced by the function.
    """
    return Image.open(path).convert("RGB")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.

    Parameters
    ----------
    path : str | Path
        Input value for ``path``.

    Returns
    -------
    result : dict[str, Any]
        Return value produced by the function.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_metrics_markdown(metrics: dict[str, Any]) -> str:
    """Format a metric dictionary as a compact Markdown table.

    Args:
        metrics: Metric mapping to render.

    Returns:
        Markdown table with one row per metric.

    Parameters
    ----------
    metrics : dict[str, Any]
        Input value for ``metrics``.

    Returns
    -------
    result : str
        Return value produced by the function.
    """
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in metrics.items():
        if isinstance(value, float):
            value_repr = f"{value:.4f}"
        else:
            value_repr = str(value)
        lines.append(f"| `{key}` | `{value_repr}` |")
    return "\n".join(lines)

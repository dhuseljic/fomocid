"""Connected-component analysis for binary magnetic domain patterns."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


@dataclass(frozen=True)
class DomainComponent:
    """Shape measurements and classification for one magnetic domain."""

    label: int
    polarity: int
    area: int
    eccentricity: float
    circularity: float
    touches_border: bool
    is_bubble: bool


def _component_perimeter(mask: NDArray[np.bool_]) -> float:
    """Return a four-neighbour digital perimeter in pixel-edge units."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return float(
        np.sum(padded[1:, :] != padded[:-1, :])
        + np.sum(padded[:, 1:] != padded[:, :-1])
    )


def _component_eccentricity(mask: NDArray[np.bool_]) -> float:
    """Estimate eccentricity from the eigenvalues of the pixel covariance."""
    coordinates = np.argwhere(mask)
    if len(coordinates) < 3:
        return 0.0
    covariance = np.cov(coordinates, rowvar=False, bias=True)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    if eigenvalues[0] <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - eigenvalues[1] / eigenvalues[0])))


def classify_magnetic_domains(
    pattern: ArrayLike,
    *,
    analysis_mask: ArrayLike | None = None,
    threshold: float = 0.0,
    min_area: int = 9,
    max_eccentricity: float = 0.8,
    min_circularity: float = 0.5,
    connectivity: int = 2,
) -> dict:
    """Segment a magnetic pattern and classify minority domains.

    The input is binarized at ``threshold``. When ``analysis_mask`` is given,
    only pixels inside that field of view contribute to fractions, components,
    and counts. Connected components are measured in both polarities, while the
    global stripe/bubble decision is based on the minority polarity: compact,
    non-boundary components are bubbles and every other resolved minority
    component is stripe-like. Components smaller than ``min_area`` are recorded
    as noise and excluded from both counts.

    A component is a bubble when it does not touch the array boundary, its
    covariance eccentricity is at most ``max_eccentricity``, and its digital
    circularity ``4*pi*area/perimeter**2`` is at least ``min_circularity``.
    This deliberately treats boundary-clipped and elongated domains as
    stripe-like because their closed circular shape cannot be established.

    Returns a dictionary containing the binary pattern, a label image for the
    minority phase, per-component measurements, counts, and an overall
    morphology label: ``uniform``, ``noise``, ``bubbles``, ``stripes``, or
    ``mixed``.
    """
    values = np.asarray(pattern)
    if values.ndim != 2:
        raise ValueError(f"pattern must be a 2-D array, got shape {values.shape}")
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if not 0 <= max_eccentricity <= 1:
        raise ValueError("max_eccentricity must be between 0 and 1")
    if min_circularity < 0:
        raise ValueError("min_circularity must be non-negative")
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")

    if analysis_mask is None:
        field_of_view = np.ones(values.shape, dtype=bool)
    else:
        field_of_view = np.asarray(analysis_mask, dtype=bool)
        if field_of_view.shape != values.shape:
            raise ValueError(
                "analysis_mask must match pattern shape; "
                f"got {field_of_view.shape} and {values.shape}"
            )
        if not np.any(field_of_view):
            raise ValueError("analysis_mask must contain at least one True pixel")

    positive = (values > threshold) & field_of_view
    positive_fraction = float(np.mean(positive[field_of_view]))
    if positive_fraction in (0.0, 1.0):
        return {
            "binary_pattern": np.where(values > threshold, 1, -1).astype(np.int8),
            "minority_polarity": -1 if positive_fraction == 1.0 else 1,
            "minority_label_image": np.zeros(values.shape, dtype=np.int32),
            "components": [],
            "minority_components": [],
            "bubble_count": 0,
            "stripe_count": 0,
            "noise_count": 0,
            "morphology": "uniform",
        }

    minority_polarity = 1 if positive_fraction <= 0.5 else -1
    structure = ndimage.generate_binary_structure(2, connectivity)
    components: list[DomainComponent] = []
    minority_labels = np.zeros(values.shape, dtype=np.int32)

    for polarity, phase_mask in (
        (1, positive),
        (-1, (~positive) & field_of_view),
    ):
        labels, count = ndimage.label(phase_mask, structure=structure)
        if polarity == minority_polarity:
            minority_labels = labels.astype(np.int32, copy=False)
        for label_index in range(1, count + 1):
            component_mask = labels == label_index
            area = int(component_mask.sum())
            rows, columns = np.nonzero(component_mask)
            touches_border = bool(
                np.any(
                    ndimage.binary_dilation(component_mask, structure=structure)
                    & ~field_of_view
                )
                or np.any(rows == 0)
                or np.any(rows == values.shape[0] - 1)
                or np.any(columns == 0)
                or np.any(columns == values.shape[1] - 1)
            )
            perimeter = _component_perimeter(component_mask)
            circularity = (
                float(4.0 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0
            )
            eccentricity = _component_eccentricity(component_mask)
            is_bubble = bool(
                area >= min_area
                and not touches_border
                and eccentricity <= max_eccentricity
                and circularity >= min_circularity
            )
            components.append(
                DomainComponent(
                    label=label_index,
                    polarity=polarity,
                    area=area,
                    eccentricity=eccentricity,
                    circularity=circularity,
                    touches_border=touches_border,
                    is_bubble=is_bubble,
                )
            )

    minority_components = [
        component for component in components if component.polarity == minority_polarity
    ]
    resolved = [component for component in minority_components if component.area >= min_area]
    bubble_count = sum(component.is_bubble for component in resolved)
    stripe_count = sum(not component.is_bubble for component in resolved)
    noise_count = len(minority_components) - len(resolved)
    if not resolved:
        morphology = "noise"
    elif bubble_count and stripe_count:
        morphology = "mixed"
    elif bubble_count:
        morphology = "bubbles"
    else:
        morphology = "stripes"

    return {
        "binary_pattern": np.where(values > threshold, 1, -1).astype(np.int8),
        "minority_polarity": minority_polarity,
        "minority_label_image": minority_labels,
        "components": [asdict(component) for component in components],
        "minority_components": [asdict(component) for component in minority_components],
        "bubble_count": int(bubble_count),
        "stripe_count": int(stripe_count),
        "noise_count": int(noise_count),
        "morphology": morphology,
    }

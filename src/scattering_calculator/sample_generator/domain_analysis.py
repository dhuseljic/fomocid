"""Connected-component analysis for binary magnetic domain patterns."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.spatial import KDTree


@dataclass(frozen=True)
class DomainComponent:
    """Shape measurements and classification for one magnetic domain."""

    label: int
    polarity: int
    area: int
    visible_area: int
    touches_aperture: bool
    eccentricity: float
    circularity: float
    touches_border: bool
    is_bubble: bool
    is_boundary_fragment: bool


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


def _unwrap_coordinates(coordinates, shape):
    """Place a compact toroidal component in a contiguous coordinate chart."""
    result = coordinates.copy()
    for axis, size in enumerate(shape):
        positions = np.unique(coordinates[:, axis])
        gaps = np.diff(np.r_[positions, positions[0] + size])
        start = positions[(np.argmax(gaps) + 1) % len(positions)]
        result[:, axis] = (coordinates[:, axis] - start) % size
    return result


def _periodic_labels(mask, structure):
    labels, count = ndimage.label(mask, structure=structure)
    parent = np.arange(count + 1)
    def root(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label
    # ndimage already joins every interior neighbor. Only seam crossings can
    # connect distinct labels; sorting all image pixels here was O(H*W log(H*W)).
    height, width = labels.shape
    rows, columns = np.arange(height), np.arange(width)
    seam_pairs = []
    for dy, dx in np.argwhere(structure) - 1:
        # One direction per undirected adjacency is sufficient.
        if dy < 0 or (dy == 0 and dx <= 0):
            continue
        if dy:
            row = 0 if dy > 0 else height - 1
            seam_pairs.append(np.column_stack((labels[row, :],
                labels[(row-dy) % height, (columns-dx) % width])))
        if dx:
            column = 0 if dx > 0 else width - 1
            seam_pairs.append(np.column_stack((labels[:, column],
                labels[(rows-dy) % height, (column-dx) % width])))
    if seam_pairs:
        pairs = np.concatenate(seam_pairs)
        pairs = pairs[np.all(pairs != 0, axis=1)]
        for a, b in np.unique(pairs, axis=0):
            parent[root(b)] = root(a)
    mapping = np.array([root(i) for i in range(count + 1)])
    return mapping[labels], count


def classify_magnetic_domains(
    pattern: ArrayLike,
    *,
    analysis_mask: ArrayLike | None = None,
    threshold: float = 0.0,
    min_area: int = 9,
    max_eccentricity: float = 0.95,
    min_circularity: float = 0.4,
    connectivity: int = 2,
    periodic: bool = False,
    boundary_context_periodic: bool = False,
) -> dict:
    """Segment and classify visible magnetic domains of both polarities.

    The input is binarized at ``threshold`` and connected components are
    segmented over the full simulation grid. ``analysis_mask`` (the OH) selects
    which domains to count: any component with at least one visible pixel is
    included once. Shape, area cutoff and order measurements use the full
    component, including pixels outside the OH. Both polarities contribute to
    the totals, including a connected majority background. Per-polarity and
    majority/minority counts are also returned. The minority polarity is
    determined over the full grid; at equal fractions it is positive.
    A globally uniform pattern counts as one stripe-like domain and is saturated.

    Compact components are bubbles when their covariance eccentricity is at
    most ``max_eccentricity`` and digital circularity
    ``4*pi*area/perimeter**2`` is at least ``min_circularity``. Clipping by the
    OH does not change this decision. With open boundaries, components touching
    the outer simulation edge remain stripe-like because their full shape is unavailable.
    With ``periodic=True`` opposite simulation edges are joined and compact
    domains are unwrapped before shape measurement. Winding stripes are not
    bubbles. Defaults admit ellipses up to about 3.2:1 axis ratio (eccentricity
    0.95), with circularity at least 0.4 to reject irregular, extended domains.
    When ``boundary_context_periodic=True`` with open counting, opposite edges
    are consulted solely to identify fragments of confirmed periodic bubbles.
    Such fragments are recorded in ``boundary_fragment_count``, excluded from
    both B and S and order measurements. This preserves open-field bubble counts
    without falsely calling clipped bubbles stripes. Use only for a periodic
    generating solver, never to assume missing context for an arbitrary image.
    Components with full area below ``min_area`` are recorded as noise.
    ``area`` is the full component area; ``visible_area`` is its OH overlap.
    ``touches_border`` refers only to the outer simulation boundary, while
    ``touches_aperture`` records extension outside the OH.

    Returns a dictionary containing the binary pattern, a label image for the
    minority phase, per-component measurements, counts, and an overall
    morphology label: ``uniform``, ``noise``, ``bubbles``, ``stripes``, or
    ``mixed``.
    State follows total counts B and S: bubble for B>0 and S<=1, mixed for
    B>0 and S>1, stripe for B=0 and S>1, saturated for B=0 and S=1.
    S=0 with B>0 can occur when the OH excludes the background. Zero resolved
    domains give ``unresolved``; ``noise_count`` records discarded islands. Stripe nematic order measures agreement of component
    major axes (one elongated component has order one). Local/global sixfold
    bond order uses only bubble centers and requires at least seven bubbles.
    These describe orientation, not translational periodicity or topology.
    """
    values = np.asarray(pattern)
    if not np.isfinite(values).all() or values.size == 0:
        raise ValueError("pattern must contain finite values and be nonempty")
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

    positive = values > threshold
    positive_fraction = float(np.mean(positive))
    minority_polarity = 1 if positive_fraction <= 0.5 else -1
    structure = ndimage.generate_binary_structure(2, connectivity)
    components: list[DomainComponent] = []
    minority_labels = np.zeros(values.shape, dtype=np.int32)
    polarity_labels = {}

    context = None
    if boundary_context_periodic and not periodic:
        # Only use this when the generating solver really has periodic edges.
        # Context determines shape; it does not join the open-field counts.
        context = classify_magnetic_domains(values, threshold=threshold, min_area=min_area,
            max_eccentricity=max_eccentricity, min_circularity=min_circularity,
            connectivity=connectivity, periodic=True)
    context_bubbles = {} if context is None else {
        polarity: {c["label"] for c in context["components"]
                   if c["polarity"] == polarity and c["is_bubble"]}
        for polarity in (1, -1)
    }
    for polarity, phase_mask in (
        (1, positive),
        (-1, ~positive),
    ):
        labels, count = (_periodic_labels(phase_mask, structure) if periodic
                         else ndimage.label(phase_mask, structure=structure))
        polarity_labels[polarity] = labels.astype(np.int32, copy=False)
        for label_index in range(1, count + 1):
            component_mask = labels == label_index
            if not np.any(component_mask):
                continue
            visible_area = int(np.count_nonzero(component_mask & field_of_view))
            if visible_area == 0:
                continue
            area = int(component_mask.sum())
            rows, columns = np.nonzero(component_mask)
            touches_border = bool(
                np.any(rows == 0)
                or np.any(rows == values.shape[0] - 1)
                or np.any(columns == 0)
                or np.any(columns == values.shape[1] - 1)
            )
            shape_mask = component_mask
            winds = False
            if periodic:
                coordinates = _unwrap_coordinates(np.argwhere(component_mask), values.shape)
                winds = any(len(np.unique(coordinates[:, axis])) == size for axis, size in enumerate(values.shape))
                shape_mask = np.zeros(tuple(coordinates.max(axis=0) + 1), dtype=bool)
                shape_mask[tuple(coordinates.T)] = True
            perimeter = _component_perimeter(shape_mask)
            circularity = (
                float(4.0 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0
            )
            eccentricity = _component_eccentricity(shape_mask)
            is_bubble = bool(
                area >= min_area
                and (not winds if periodic else not touches_border)
                and eccentricity <= max_eccentricity
                and circularity >= min_circularity
            )
            boundary_fragment = False
            if context is not None and touches_border and not is_bubble:
                context_labels = context["positive_label_image" if polarity == 1 else "negative_label_image"]
                boundary_fragment = int(context_labels[rows[0], columns[0]]) in context_bubbles[polarity]
            components.append(
                DomainComponent(
                    label=label_index,
                    polarity=polarity,
                    area=area,
                    visible_area=visible_area,
                    touches_aperture=visible_area < area,
                    eccentricity=eccentricity,
                    circularity=circularity,
                    touches_border=touches_border,
                    is_bubble=is_bubble,
                    is_boundary_fragment=boundary_fragment,
                )
            )

    minority_components = [
        component for component in components if component.polarity == minority_polarity
    ]
    # Retain full component geometry, but remove labels with no OH intersection.
    for polarity in (1, -1):
        visible_labels = [c.label for c in components if c.polarity == polarity]
        polarity_labels[polarity] = np.where(
            np.isin(polarity_labels[polarity], visible_labels), polarity_labels[polarity], 0
        ).astype(np.int32)
    minority_labels = polarity_labels[minority_polarity]

    return _state_descriptors({
        "binary_pattern": np.where(values > threshold, 1, -1).astype(np.int8),
        "minority_polarity": minority_polarity,
        "minority_label_image": minority_labels,
        "positive_label_image": polarity_labels[1],
        "negative_label_image": polarity_labels[-1],
        "components": [asdict(component) for component in components],
        "minority_components": [asdict(component) for component in minority_components],

        "periodic": periodic,
        "boundary_context_periodic": boundary_context_periodic,
        "is_uniform": positive_fraction in (0.0, 1.0),
    }, min_area, threshold, max_eccentricity, min_circularity, connectivity)


def _state_descriptors(result, min_area, threshold, max_eccentricity, min_circularity, connectivity):
    """Add reproducible state and orientational order measurements."""
    for prefix, polarity in (("positive", 1), ("negative", -1),
                             ("minority", result["minority_polarity"]),
                             ("majority", -result["minority_polarity"])):
        phase = [c for c in result["components"] if c["polarity"] == polarity]
        resolved = [c for c in phase if c["area"] >= min_area or result["is_uniform"]]
        result[f"{prefix}_bubble_count"] = sum(c["is_bubble"] for c in resolved)
        result[f"{prefix}_stripe_count"] = sum(not c["is_bubble"] and not c["is_boundary_fragment"] for c in resolved)
        result[f"{prefix}_boundary_fragment_count"] = sum(c["is_boundary_fragment"] for c in resolved)
        result[f"{prefix}_noise_count"] = len(phase) - len(resolved)
    for kind in ("bubble", "stripe", "noise", "boundary_fragment"):
        result[f"{kind}_count"] = result[f"positive_{kind}_count"] + result[f"negative_{kind}_count"]
    bubbles, stripes = result["bubble_count"], result["stripe_count"]
    result["morphology"] = ("uniform" if result["is_uniform"] else "mixed" if bubbles and stripes else "bubbles" if bubbles else
                            "stripes" if stripes else "noise" if result["noise_count"] else "uniform")
    labels = result["minority_label_image"]
    stripe_vectors, centers = [], []
    for component in result["components"]:
        labels = result["positive_label_image"] if component["polarity"] == 1 else result["negative_label_image"]
        if component["area"] < min_area or component["is_boundary_fragment"]:
            continue
        coordinates = np.argwhere(labels == component["label"])
        original_coordinates = coordinates
        if result["periodic"]:
            coordinates = _unwrap_coordinates(coordinates, labels.shape)
        if component["is_bubble"]:
            if result["periodic"]:
                # Circular centroids retain positions on the original torus.
                angles = original_coordinates * (2 * np.pi / np.array(labels.shape))
                center = (np.angle(np.mean(np.exp(1j * angles), axis=0)) % (2*np.pi)) * np.array(labels.shape) / (2*np.pi)
                centers.append(center)
            else:
                centers.append(coordinates.mean(axis=0))
        elif len(coordinates) >= 3:
            covariance = np.cov(coordinates, rowvar=False, bias=True)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            # Isotropic shapes have no reliable major axis. Weight elongated
            # domains by area and anisotropy so arbitrary axes do not dominate.
            weight = component["area"] * (eigenvalues[1] - eigenvalues[0]) / max(eigenvalues.sum(), 1e-15)
            axis = eigenvectors[:, 1]
            stripe_vectors.append((weight, np.exp(2j * np.arctan2(axis[0], axis[1]))))
    weight_sum = sum(w for w, _ in stripe_vectors)
    nematic = float(np.clip(abs(sum(w * v for w, v in stripe_vectors)) / weight_sum, 0, 1)) if weight_sum > 1e-12 else None
    local_sixfold = global_sixfold = None
    if len(centers) >= 7:
        points = np.asarray(centers)
        _, neighbors = KDTree(points, boxsize=labels.shape if result["periodic"] else None).query(points, k=7)
        vectors = points[neighbors[:, 1:]] - points[:, None, :]
        if result["periodic"]:
            size = np.array(labels.shape)
            vectors = (vectors + size / 2) % size - size / 2
        angles = np.arctan2(vectors[..., 0], vectors[..., 1])
        psi6 = np.mean(np.exp(6j * angles), axis=1)
        local_sixfold = float(np.clip(np.mean(np.abs(psi6)), 0, 1))
        global_sixfold = float(np.clip(abs(np.mean(psi6)), 0, 1))
    result.update(
        state=("bubble" if stripes <= 1 else "mixed") if bubbles else (
            "saturated" if stripes == 1 else "stripe" if stripes > 1 else "unresolved"),
        stripe_nematic_order=nematic,
        local_sixfold_order=local_sixfold,
        sixfold_order=global_sixfold,
        interior_bubble_count=len(centers),
        classification_min_area_px=int(min_area), classification_threshold=float(threshold),
        bubble_max_eccentricity=float(max_eccentricity), bubble_min_circularity=float(min_circularity),
        classification_connectivity=int(connectivity),
    )
    return result


def write_domain_classification(group, pattern, **classification_options):
    """Write scalar state descriptors into an existing HDF5 group.

    Undefined order parameters are saved as NaN. The caller owns the file and
    chooses the group path; arrays and component tables are not duplicated.
    """
    result = classify_magnetic_domains(pattern, **classification_options)
    group.attrs["method"] = "both_polarities_connected_components_v6"
    group.attrs["boundary_convention"] = "full simulation shapes; count OH intersections; " + ("periodic outer grid" if result["periodic"] else "open outer grid")
    for name, value in result.items():
        if name in {"binary_pattern", "minority_label_image", "positive_label_image", "negative_label_image", "components", "minority_components"}:
            continue
        group.create_dataset(name, data=np.nan if value is None else value)
    return result

"""Sampling helpers for morphology-balanced binary-domain data sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import fill_small_domain_holes
from .domain_analysis import classify_magnetic_domains
from .gray_scott_generator_binary import generate


@dataclass(frozen=True)
class DomainPhaseSpace:
    """A classified regular grid in ``(k0, eps, target_mean)`` space."""

    k0_values: np.ndarray
    eps_values: np.ndarray
    target_mean_values: np.ndarray
    stripe_points: np.ndarray
    bubble_points: np.ndarray


def load_or_analyze_domain_phase_space(
    cache_path,
    *,
    k0_values,
    eps_values,
    target_mean_values,
    shape=(128, 128),
    n_steps=80,
    seed=7,
    min_area=9,
    max_eccentricity=0.85,
    min_circularity=0.45,
    max_hole_area=9,
    verbose=False,
):
    """Load a matching phase-space cache or analyze and save it once."""
    cache_path = Path(cache_path)
    axes = [np.asarray(values, dtype=float) for values in (
        k0_values, eps_values, target_mean_values
    )]
    settings = {
        "shape": np.asarray(shape, dtype=int),
        "n_steps": int(n_steps),
        "seed": int(seed),
        "min_area": int(min_area),
        "max_eccentricity": float(max_eccentricity),
        "min_circularity": float(min_circularity),
        "max_hole_area": int(max_hole_area),
    }
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                matches = (
                    np.array_equal(cached["k0_values"], axes[0])
                    and np.array_equal(cached["eps_values"], axes[1])
                    and np.array_equal(cached["target_mean_values"], axes[2])
                    and all(np.array_equal(cached[key], value)
                            for key, value in settings.items())
                )
                if matches:
                    if verbose:
                        print(f"Loading cached domain phase space: {cache_path}")
                    return DomainPhaseSpace(
                        cached["k0_values"].copy(),
                        cached["eps_values"].copy(),
                        cached["target_mean_values"].copy(),
                        cached["stripe_points"].copy(),
                        cached["bubble_points"].copy(),
                    )
        except (KeyError, OSError, ValueError):
            pass

    if verbose:
        print(f"Analyzing domain phase space once; cache will be: {cache_path}")
    phase_space = analyze_domain_phase_space(
        k0_values=axes[0], eps_values=axes[1], target_mean_values=axes[2],
        shape=shape, n_steps=n_steps, seed=seed, min_area=min_area,
        max_eccentricity=max_eccentricity,
        min_circularity=min_circularity, max_hole_area=max_hole_area,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        k0_values=phase_space.k0_values,
        eps_values=phase_space.eps_values,
        target_mean_values=phase_space.target_mean_values,
        stripe_points=phase_space.stripe_points,
        bubble_points=phase_space.bubble_points,
        **settings,
    )
    if verbose:
        print(f"Saved domain phase-space cache: {cache_path}")
    return phase_space


def analyze_domain_phase_space(
    *,
    k0_values,
    eps_values,
    target_mean_values,
    shape=(128, 128),
    n_steps=80,
    seed=7,
    min_area=9,
    max_eccentricity=0.85,
    min_circularity=0.45,
    max_hole_area=9,
) -> DomainPhaseSpace:
    """Classify a parameter grid into stripe-rich and bubble-rich points.

    A mixed point is assigned to the richer class by comparing its resolved
    stripe and bubble counts. Ties and saturated/noise-only points are omitted.
    """
    axes = [np.asarray(values, dtype=float) for values in (
        k0_values, eps_values, target_mean_values
    )]
    stripe_points, bubble_points = [], []
    for target_mean in axes[2]:
        for eps in axes[1]:
            for k0 in axes[0]:
                _, binary, _, _ = generate(
                    batch=1, H=shape[0], W=shape[1], n_steps=n_steps,
                    region="custom", use_gpu=False, seed=seed, k0=k0,
                    eps=eps, target_mean=target_mean, noise_amp=0.0,
                    quadratic_coefficient=0.0,
                )
                pattern = fill_small_domain_holes(
                    np.asarray(binary[0], dtype=float), max_hole_area
                )
                result = classify_magnetic_domains(
                    pattern, min_area=min_area,
                    max_eccentricity=max_eccentricity,
                    min_circularity=min_circularity,
                )
                point = (k0, eps, target_mean)
                if result["stripe_count"] > result["bubble_count"]:
                    stripe_points.append(point)
                elif result["bubble_count"] > result["stripe_count"]:
                    bubble_points.append(point)
    if not stripe_points or not bubble_points:
        raise RuntimeError("Phase-space scan did not find both morphology classes")
    return DomainPhaseSpace(
        axes[0], axes[1], axes[2],
        np.asarray(stripe_points), np.asarray(bubble_points),
    )


def sample_morphology_region(phase_space, morphology, rng):
    """Sample continuously around a grid point belonging to one rich region."""
    points = {
        "stripe": phase_space.stripe_points,
        "bubble": phase_space.bubble_points,
    }[morphology]
    centre = points[rng.integers(len(points))]
    sampled = []
    for value, axis in zip(centre, (
        phase_space.k0_values,
        phase_space.eps_values,
        phase_space.target_mean_values,
    )):
        index = int(np.argmin(np.abs(axis - value)))
        low = axis[index] if index == 0 else 0.5 * (axis[index - 1] + value)
        high = axis[index] if index == len(axis) - 1 else 0.5 * (value + axis[index + 1])
        sampled.append(float(rng.uniform(low, high)) if high > low else float(value))
    return dict(zip(("k0", "eps", "target_mean"), sampled))


def sample_verified_morphology_region(
    phase_space,
    morphology,
    rng,
    *,
    shape=(128, 128),
    n_steps=80,
    seed=7,
    min_area=9,
    max_eccentricity=0.85,
    min_circularity=0.45,
    max_hole_area=9,
    max_attempts=100,
    coordinate_bounds=None,
):
    """Draw off-grid coordinates and verify that their requested class wins."""
    for _ in range(max_attempts):
        point = sample_morphology_region(phase_space, morphology, rng)
        if coordinate_bounds is not None and any(
            not coordinate_bounds[name][0] <= point[name] <= coordinate_bounds[name][1]
            for name in ("k0", "eps", "target_mean")
        ):
            continue
        _, binary, _, _ = generate(
            batch=1, H=shape[0], W=shape[1], n_steps=n_steps,
            region="custom", use_gpu=False, seed=seed, noise_amp=0.0,
            quadratic_coefficient=0.0, **point,
        )
        pattern = fill_small_domain_holes(
            np.asarray(binary[0], dtype=float), max_hole_area
        )
        result = classify_magnetic_domains(
            pattern, min_area=min_area, max_eccentricity=max_eccentricity,
            min_circularity=min_circularity,
        )
        if morphology == "stripe":
            accepted = result["stripe_count"] > result["bubble_count"]
        else:
            accepted = result["bubble_count"] > result["stripe_count"]
        if accepted:
            return point
    raise RuntimeError(
        f"Could not verify an off-grid {morphology}-rich coordinate in "
        f"{max_attempts} attempts"
    )


def balanced_state_schedule(n_samples, rng):
    """Return shuffled state targets whose class counts differ by at most one."""
    states = np.asarray(["saturated", "stripe", "bubble"] * ((n_samples + 2) // 3))[
        :n_samples
    ]
    rng.shuffle(states)
    return states.tolist()

"""Energy draws use the absorption maximum and a genuinely truncated Gaussian."""
import numpy as np
import pytest

from scattering_calculator.simulation_pipelines.energy_sampling import (
    co_absorption_peak_energy, sample_absorption_edge_energy,
)


def test_peak_and_truncated_gaussian_distribution():
    peak = co_absorption_peak_energy()
    assert peak == pytest.approx(781.1386599819, abs=1e-6)
    rng = np.random.default_rng(42)
    offsets = np.array([sample_absorption_edge_energy(rng)-peak for _ in range(20000)])
    assert np.all(np.abs(offsets) < 2)
    assert abs(offsets.mean()) < .025
    # Conditioning a sigma=1 normal on +/-2 reduces its output std to ~0.880.
    assert .865 < offsets.std() < .895
    first, second = np.random.default_rng(3), np.random.default_rng(3)
    assert [sample_absorption_edge_energy(first) for _ in range(20)] == [sample_absorption_edge_energy(second) for _ in range(20)]


def test_tail_is_redrawn_instead_of_clipped():
    class Draws:
        values = iter([3., -2.5, .75])
        def normal(self, mean, sigma):
            assert (mean, sigma) == (0., 1.)
            return next(self.values)
    assert sample_absorption_edge_energy(Draws()) == co_absorption_peak_energy()+.75

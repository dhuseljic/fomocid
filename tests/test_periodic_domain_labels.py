"""Periodic seam optimization preserves connectivity and component diagnostics."""
import json

import numpy as np
import pytest
from scipy import ndimage

from scattering_calculator.sample_generator import domain_analysis
from scattering_calculator.sample_generator.domain_catalogue import measure_domain_pattern


def _reference_periodic_labels(mask, structure):
    """Original whole-image union algorithm used as a small-grid oracle."""
    labels, count = ndimage.label(mask, structure=structure)
    parent = np.arange(count + 1)
    def root(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label
    for dy, dx in np.argwhere(structure) - 1:
        if dy == dx == 0:
            continue
        shifted = np.roll(labels, (int(dy), int(dx)), axis=(0, 1))
        pairs = np.unique(np.stack([labels[mask], shifted[mask]], axis=1), axis=0)
        for a, b in pairs:
            if a and b:
                parent[root(b)] = root(a)
    mapping = np.array([root(i) for i in range(count + 1)])
    return mapping[labels], count


@pytest.mark.parametrize('connectivity', [1, 2])
def test_seam_only_labels_match_every_three_by_three_binary_pattern(connectivity):
    structure = ndimage.generate_binary_structure(2, connectivity)
    for bits in range(512):
        mask = ((bits >> np.arange(9)) & 1).reshape(3, 3).astype(bool)
        expected, _ = _reference_periodic_labels(mask, structure)
        actual, _ = domain_analysis._periodic_labels(mask, structure)
        pairs = np.unique(np.column_stack((expected.ravel(), actual.ravel())), axis=0)
        assert len(pairs) == len(np.unique(actual)) == len(np.unique(expected))
        np.testing.assert_array_equal(actual == 0, ~mask)


def test_precomputed_classification_preserves_all_scalar_measurements(monkeypatch):
    yy, xx = np.indices((96, 128))
    pattern = np.where(np.cos(xx*.3) + np.sin(yy*.2) > .2, 1., -1.)
    mask = (yy-48)**2 + (xx-64)**2 < 35**2
    classification = domain_analysis.classify_magnetic_domains(pattern, analysis_mask=mask, periodic=True)
    expected = measure_domain_pattern(pattern, analysis_mask=mask, periodic=True)
    import scattering_calculator.sample_generator.domain_catalogue as catalogue
    def unexpected_call(*args, **kwargs):
        raise AssertionError('Classification should not run again')
    monkeypatch.setattr(catalogue, 'classify_magnetic_domains', unexpected_call)
    assert measure_domain_pattern(pattern, classification=classification) == expected
    with pytest.raises(ValueError, match='not both'):
        measure_domain_pattern(pattern, classification=classification, periodic=True)

"""Tests for Mumax/OOMMF OVF readers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.utils.mumax import read_mumax_ovf


def _write_binary4_ovf(path: Path, data: np.ndarray) -> None:
    znodes, ynodes, xnodes, valuedim = data.shape
    assert valuedim == 3
    header = f"""# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: m
# meshtype: rectangular
# meshunit: m
# xmin: 0
# ymin: 0
# zmin: 0
# xmax: {xnodes * 2e-9}
# ymax: {ynodes * 3e-9}
# zmax: {znodes * 4e-9}
# valuedim: 3
# valuelabels: m_x m_y m_z
# valueunits: 1 1 1
# xnodes: {xnodes}
# ynodes: {ynodes}
# znodes: {znodes}
# xstepsize: 2e-09
# ystepsize: 3e-09
# zstepsize: 4e-09
# End: Header
# Begin: Data Binary 4
"""
    payload = np.concatenate(
        [np.array([1234567.0], dtype="<f4"), data.astype("<f4").ravel()]
    )
    with path.open("wb") as file_obj:
        file_obj.write(header.encode("ascii"))
        payload.tofile(file_obj)


def test_read_mumax_ovf_returns_memmap_without_copying(tmp_path: Path) -> None:
    data = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    path = tmp_path / "0000.ovf"
    _write_binary4_ovf(path, data)

    ovf = read_mumax_ovf(path)

    assert isinstance(ovf.magnetization, np.memmap)
    assert ovf.magnetization.shape == (2, 3, 4, 3)
    assert ovf.xnodes == 4
    assert ovf.ynodes == 3
    assert ovf.znodes == 2
    assert ovf.xstepsize == 2e-9
    assert ovf.dx == ovf.xstepsize
    assert ovf.dy == ovf.ystepsize
    assert ovf.dz == ovf.zstepsize
    assert ovf.size_x == pytest.approx(8e-9)
    assert ovf.size_y == pytest.approx(9e-9)
    assert ovf.size_z == pytest.approx(8e-9)
    np.testing.assert_allclose(ovf.magnetization, data)


def test_read_mumax_ovf_can_copy_into_memory(tmp_path: Path) -> None:
    data = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
    path = tmp_path / "copy.ovf"
    _write_binary4_ovf(path, data)

    ovf = read_mumax_ovf(path, mmap=False)

    assert not isinstance(ovf.magnetization, np.memmap)
    np.testing.assert_allclose(ovf.magnetization, data)

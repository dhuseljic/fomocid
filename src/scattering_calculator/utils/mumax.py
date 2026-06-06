"""Fast readers for Mumax/OOMMF OVF vector fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MumaxOVF:
    """Mumax/OOMMF OVF vector field and mesh metadata.

    ``magnetization`` is shaped ``(znodes, ynodes, xnodes, 3)`` so a z slice is
    accessed as ``magnetization[iz]`` and components are ``[..., 0:3]`` for
    ``mx, my, mz``.
    """

    path: Path
    header: dict[str, Any]
    magnetization: np.ndarray
    data_format: str
    data_offset: int

    @property
    def znodes(self) -> int:
        return int(self.header["znodes"])

    @property
    def ynodes(self) -> int:
        return int(self.header["ynodes"])

    @property
    def xnodes(self) -> int:
        return int(self.header["xnodes"])

    @property
    def zstepsize(self) -> float:
        return float(self.header["zstepsize"])

    @property
    def ystepsize(self) -> float:
        return float(self.header["ystepsize"])

    @property
    def xstepsize(self) -> float:
        return float(self.header["xstepsize"])

    @property
    def dx(self) -> float:
        return self.xstepsize

    @property
    def dy(self) -> float:
        return self.ystepsize

    @property
    def dz(self) -> float:
        return self.zstepsize

    @property
    def size_x(self) -> float:
        return float(self.header["xmax"]) - float(self.header["xmin"])

    @property
    def size_y(self) -> float:
        return float(self.header["ymax"]) - float(self.header["ymin"])

    @property
    def size_z(self) -> float:
        return float(self.header["zmax"]) - float(self.header["zmin"])

    @property
    def extent_xy(self) -> list[float]:
        return [
            float(self.header["xmin"]),
            float(self.header["xmax"]),
            float(self.header["ymin"]),
            float(self.header["ymax"]),
        ]


def _parse_header_value(value: str) -> Any:
    value = value.strip()
    if len(value.split()) > 1:
        return value
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_ovf_header(path: str | Path) -> tuple[dict[str, Any], str, int]:
    """Read an OVF header and return ``(header, data_format, data_offset)``."""
    path = Path(path)
    header: dict[str, Any] = {}
    version = None
    with path.open("rb") as file_obj:
        while True:
            line_bytes = file_obj.readline()
            if not line_bytes:
                raise ValueError(f"Reached end of {path} before '# Begin: Data ...'.")
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if "OOMMF: rectangular mesh v1.0" in line:
                version = "1.0"
            elif "OOMMF OVF 2.0" in line:
                version = "2.0"
            if line.startswith("# Begin: Data"):
                data_format = line.replace("# Begin: Data", "").strip()
                header["ovf_version"] = version
                return header, data_format, file_obj.tell()
            if not line.startswith("#") or ":" not in line:
                continue
            key, value = line[1:].split(":", 1)
            key = key.strip().lower()
            if key in {"begin", "end"}:
                continue
            header[key] = _parse_header_value(value)


def _validate_header(header: dict[str, Any]) -> None:
    required = [
        "xnodes",
        "ynodes",
        "znodes",
        "xstepsize",
        "ystepsize",
        "zstepsize",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "valuedim",
    ]
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(f"Missing OVF header fields: {missing}")
    if int(header["valuedim"]) != 3:
        raise ValueError(f"Expected valuedim=3 for mx,my,mz, got {header['valuedim']!r}.")


def _binary_format(data_format: str, version: str | None) -> tuple[float, np.dtype]:
    if data_format == "Binary 4":
        check_value = 1234567.0
        # OOMMF v1 writes big-endian; OVF 2/Mumax writes little-endian.
        dtype = np.dtype(">f4" if version == "1.0" else "<f4")
    elif data_format == "Binary 8":
        check_value = 123456789012345.0
        dtype = np.dtype(">f8" if version == "1.0" else "<f8")
    else:
        raise ValueError(
            f"Unsupported OVF data format {data_format!r}. "
            "This reader supports Binary 4 and Binary 8."
        )
    return check_value, dtype


def read_mumax_ovf(path: str | Path, *, mmap: bool = True) -> MumaxOVF:
    """Read a Mumax/OOMMF OVF vector field.

    Parameters
    ----------
    path : str or Path
        OVF file path.
    mmap : bool
        If ``True`` (default), return a lazy ``np.memmap`` view of the binary
        vector field. This makes the load/header cell effectively instant even
        for large files. If ``False``, read the full field into RAM as a NumPy
        array.
    """
    path = Path(path)
    header, data_format, data_offset = read_ovf_header(path)
    _validate_header(header)

    check_value, dtype = _binary_format(data_format, header.get("ovf_version"))
    with path.open("rb") as file_obj:
        file_obj.seek(data_offset)
        observed = np.fromfile(file_obj, dtype=dtype, count=1)
    if observed.size != 1:
        raise ValueError(f"Could not read OVF binary check value from {path}.")
    tolerance = max(1e-2, abs(check_value) * 1e-12)
    if not np.isclose(float(observed[0]), check_value, rtol=0, atol=tolerance):
        raise ValueError(
            f"Binary check value is {float(observed[0])}, not {check_value}. "
            "The file may use an unsupported endian/format combination."
        )

    shape = (
        int(header["znodes"]),
        int(header["ynodes"]),
        int(header["xnodes"]),
        3,
    )
    value_count = int(np.prod(shape))
    value_offset = data_offset + dtype.itemsize
    available_bytes = path.stat().st_size - value_offset
    required_bytes = value_count * dtype.itemsize
    if available_bytes < required_bytes:
        raise ValueError(
            f"OVF binary payload is too short: need {required_bytes} bytes, "
            f"but only {available_bytes} bytes are available."
        )

    if mmap:
        magnetization = np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=value_offset,
            shape=shape,
            order="C",
        )
    else:
        with path.open("rb") as file_obj:
            file_obj.seek(value_offset)
            magnetization = np.fromfile(
                file_obj,
                dtype=dtype,
                count=value_count,
            ).reshape(shape)

    return MumaxOVF(
        path=path,
        header=header,
        magnetization=magnetization,
        data_format=data_format,
        data_offset=value_offset,
    )


def list_ovf_files(folder: str | Path) -> list[Path]:
    """Return OVF files in ``folder`` sorted by filename."""
    folder = Path(folder)
    return sorted(folder.glob("*.ovf")) + sorted(folder.glob("*.OVF"))

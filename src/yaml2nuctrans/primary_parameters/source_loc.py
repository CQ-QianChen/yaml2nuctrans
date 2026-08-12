from dataclasses import dataclass

Coordinate1D = float
Coordinate2D = tuple[float, float]
Coordinate3D = tuple[float, float, float]


@dataclass
class PointSource1D:
    loc: Coordinate1D


@dataclass
class LineSource1D:
    start_loc: Coordinate1D
    end_loc: Coordinate1D


@dataclass
class PointSource2D:
    loc: Coordinate2D


@dataclass
class LineSource2D:
    start_loc: Coordinate2D
    end_loc: Coordinate2D


@dataclass
class SurfaceSource2D:
    """A 2D source region represented by polygon vertices."""
    vertices: list[Coordinate2D]


@dataclass
class PointSource3D:
    loc: Coordinate3D


@dataclass
class LineSource3D:
    start: Coordinate3D
    end: Coordinate3D


@dataclass
class SurfaceSource3D:
    """A 3D surface represented by three or more vertices."""
    vertices: list[Coordinate3D]


@dataclass
class VolumeSource3D:
    """A 3D volume, initially represented as an axis-aligned box."""
    min_corner: Coordinate3D
    max_corner: Coordinate3D
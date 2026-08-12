from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Geometry:
    """Store interface elevation data for one interface.
    """

    interface_name: str
    property_type: Literal["scalar"]
    source: str
    value: float
    value_min: float
    value_max: float
    value_std: float
    sample_size: int
    sampled_data: str
    unit_str: str | None
    unit_base: tuple[int, ...]
    description: str
    property_id: str
    run_value: float | None = None

    @property
    def value_for_run(self):
        """Return the elevation used in the current simulation.

        Returns:
            float: Sampled/overridden value when available; otherwise
            the central reference value.
        """
        return self.value if self.run_value is None else self.run_value

    @classmethod
    def from_entry(cls, interface_name, entry):
        """Create geometry data from one YAML entry.

        Args:
            interface_name (str): Mapping key of the entry in the
                YAML file.
            entry (dict): Mapping with the keys ``source``, ``type``,
                ``value``, ``value_min``, ``value_max``, ``value_std``,
                ``sample_size``, ``sampled_data``, ``unit_str``,
                ``unit_base``, ``description`` and ``tag: ID``.

        Returns:
            Geometry: Elevation data for the interface.
        """
        return cls(
            interface_name=str(interface_name),
            property_type=entry["type"],
            source=entry["source"],
            value=float(entry["value"]),
            value_min=float(entry["value_min"]),
            value_max=float(entry["value_max"]),
            value_std=float(entry["value_std"]),
            sample_size=int(entry['probability_distribution']["sample_size"]),
            sampled_data=str(entry['probability_distribution']["sampled_data"]),
            unit_str=entry.get("unit_str"),
            unit_base=tuple(entry["unit_base"]),
            description=entry["description"],
            property_id=entry["tag"]["ID"],
        )


@dataclass
class GeometryData:
    """Store interface elevation data for all interfaces.
    """

    by_rock_interface: dict[str, Geometry]

    @classmethod
    def from_folder(cls, geometry_data_path):
        """Create geometry data for all interfaces from a YAML file.

        Args:
            geometry_data_path (str or pathlib.Path): YAML path containing a
                mapping of interface names to entries.

        Returns:
            GeometryData: Elevation data of all interfaces.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        geometry_data_path = Path(geometry_data_path)
        geometry_file_path = sorted(geometry_data_path.glob("*.yaml"))[0]

        if not geometry_file_path.is_file():
            raise FileNotFoundError(
                f"Geometry YAML file was not found: {geometry_file_path}"
            )

        with open(geometry_file_path, "r", encoding="utf-8") as f:
                geometry_data = yaml.safe_load(f)

        all_geometry_data = {
            str(interface_name): Geometry.from_entry(interface_name, entry[0])
            for interface_name, entry in geometry_data.items()
        }

        return cls(by_rock_interface=all_geometry_data)

    @property
    def interfaces(self):
        """Return the elevations used in the current simulation.

        Returns:
            dict[str, float]: ``value_for_run`` of each interface,
            indexed by interface name.
        """
        return {
            interface_name: geometry.value_for_run
            for interface_name, geometry in self.by_rock_interface.items()
        }


@dataclass
class Units1D:
    """Store interface elevations for a one-dimensional geological model.

    Attributes:
        interfaces (dict[str, float]): Interface elevations indexed by
            interface name, for example `Quaternary_top`.
    """

    interfaces: dict[str, float]

    @classmethod
    def from_geometry_data(cls, geometry_data):
        """Create unit interfaces from a YAML file.

        Args:
            geometry_data (GeometryData): Geometry data for all interfaces.

        Returns:
            Units1D: One-dimensional interface data.
        """

        return Units1D(interfaces=geometry_data.interfaces)
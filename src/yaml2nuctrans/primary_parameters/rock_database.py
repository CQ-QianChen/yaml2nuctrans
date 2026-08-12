from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class RockProperty:
    """One physical property of one rock unit."""

    property_name: str
    property_type: Literal["scalar"]
    source: str
    value: float
    value_min: float
    value_max: float
    value_std: float
    sample_size: int
    sampled_data: str
    unit_str: str | None
    unit_base: tuple[int, int, int, int, int, int, int]
    description: str
    property_id: str
    run_value: float | None = None

    @property
    def value_for_run(self):
        """Value used by the simulation.

        Returns:
            float: return value for the simulation.
        """
        
        return self.value if self.run_value is None else self.run_value


@dataclass
class RockUnit:
    """All physical properties belonging to one rock unit."""

    rock_name: str
    simplified_lithology: list[str]
    properties: dict[str, RockProperty]

    def get_property(self, property_name):
        """Return a rock property by its YAML property name

        Args:
            property_name (str): property name

        Raises:
            KeyError: Property cannot be found.

        Returns:
            RockProperty: One physical property of one rock unit.
        """
        
        try:
            return self.properties[property_name]
        except KeyError as error:
            available = ", ".join(self.properties)
            raise KeyError(
                f"Property '{property_name}' not found for '{self.rock_name}'. "
                f"Available: {available}"
            ) from error

    @classmethod
    def from_yaml(cls, filename, simplified_lithology, rock_name=None):
        """Create a RockUnit from one rock-property YAML file

        Args:
            filename (str): path to the rock-property YAML file
            simplified_lithology (list[str]): Lithology classes
            associated with the unit.
            rock_name (str, optional): rock unit name. Defaults to None.

        Returns:
            RockUnit: All physical properties belonging to one rock unit
        """
        path = Path(filename)

        with path.open(encoding="utf-8") as file:
            raw_properties = yaml.safe_load(file)

        properties = {}

        for property_name, data_list in raw_properties.items():
            rock_property_data = data_list[0]
            distribution = rock_property_data.get("probability_distribution", {})
            tag = rock_property_data.get("tag", {})

            properties[property_name] = RockProperty(
                property_name=property_name,
                property_type=rock_property_data["type"],
                source=rock_property_data["source"],
                value=float(rock_property_data["value"]),
                value_min=float(rock_property_data["value_min"]),
                value_max=float(rock_property_data["value_max"]),
                value_std=float(rock_property_data["value_std"]),
                sample_size=int(distribution["sample_size"]),
                sampled_data=distribution["sampled_data"],
                unit_str=rock_property_data.get("unit_str"),
                unit_base=tuple(rock_property_data["unit_base"]),
                description=rock_property_data["description"],
                property_id=tag["ID"],
            )

        return cls(
            rock_name=rock_name or path.stem,
            simplified_lithology=simplified_lithology,
            properties=properties,
        )


@dataclass
class RockData:
    """All rock-unit data loaded from a folder of YAML files."""

    by_rock_name: dict[str, RockUnit]

    @classmethod
    def from_folder(cls, rock_data_folder, site_data_folder, recursive=False):
        """Load every YAML rock-property file in a folder.

        Args:
            rock_data_folder (str): path to the rock data folder
            site_data_folder (str): path to the site folder containing `simplified_lithology` YAML entries.
            recursive (bool, optional): decides whether to search through subfolders or only the current folder. Defaults to False.

        Raises:
            FileNotFoundError: Rock-data folder does not exist or is not a directory

        Returns:
            RockDatabase: All rock-unit data and lithology loaded YAML files .
        """
        
        rock_data_folder = Path(rock_data_folder)

        if not rock_data_folder.is_dir():
            raise FileNotFoundError(
                f"Rock-data folder does not exist {rock_data_folder}"
            )

        lithology_by_unit = cls._load_lithology_data(site_data_folder)

        finder = rock_data_folder.rglob if recursive else rock_data_folder.glob

        yaml_files = sorted(
            [
                *finder("*.yaml"),
                *finder("*.yml"),
            ]
        )

        rocks: dict[str, RockUnit] = {}

        for yaml_file in yaml_files:
            path = Path(yaml_file)
            rocks[path.stem] = RockUnit.from_yaml(
                filename=yaml_file,
                simplified_lithology=lithology_by_unit.get(path.stem),
            )

        return cls(by_rock_name=rocks)

    @staticmethod
    def _load_lithology_data(site_folder):
        """Load simplified lithology lists from a site YAML.

        Args:
            site_folder (str or pathlib.Path): Site folder containing unit lithology information.

        Returns:
            dict[str, list[str]]: Simplified lithology classes indexed by rock name.
        """

        site_folder = Path(site_folder)
        site_file = sorted(site_folder.glob("*.yaml"))[0]

        with site_file.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)

        lithology_by_rock = {}

        for rock_name, lithology_data in data.items():
            if not isinstance(lithology_data, dict):
                continue

            simplified_lithology = lithology_data.get(
                "simplified_lithology"
            )


            lithology_by_rock[rock_name] = simplified_lithology

        return lithology_by_rock
   
    def get(self, rock_name):
        """Return one rock unit by the rock name.

        Args:
            rock_name (str): name of the rock unit.

        Raises:
            KeyError: Rock unit cannot be found.

        Returns:
            RockUnit: All physical properties in one rock unit.
        """
        
        try:
            return self.by_rock_name[rock_name]
        except KeyError as error:
            available = ", ".join(self.by_rock_name)
            raise KeyError(
                f"Rock unit '{rock_name}' was not found. "
                f"Available: {available}"
            ) from error

    @property
    def rock_names(self):
        """_Names of all loaded rock units.

        Returns:
            list: a list of rock names
        """
        
        return list(self.by_rock_name)
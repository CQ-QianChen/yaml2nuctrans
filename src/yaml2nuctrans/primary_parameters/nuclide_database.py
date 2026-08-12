from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import re

import yaml


@dataclass
class SpeciesType:
    """Store chemical-species information for one nuclide.
    """

    source: str
    species_type: str
    element_category: str
    description: str


@dataclass
class EmittedEnergy:
    """Store emitted energy data for one nuclide.
    """

    source: str
    alpha: float
    electron: float
    photon: float
    total: float
    unit_str: str | None
    unit_base: tuple[int, ...]


@dataclass
class NuclidePropertyData:
    """Store nuclide property data for one nuclide in one rock unit.
    """
    nuclide_name: str
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
        """Return the value used in the current simulation.

        Returns:
            float: Sampled/overridden value when available; otherwise
            the central reference value.
        """
        return self.value if self.run_value is None else self.run_value

@dataclass
class Nuclide:
    """Store all available data for one nuclide.

    Attributes:
        nuclide_name (str): Nuclide name, for example "I-129".
        species (SpeciesType or None): Chemical-species information.
        emitted_energy (EmittedEnergy or None): Emitted-energy data.
        sorption_by_rock (dict[str, NuclidePropertyData]):
                    Unit-specific sorption coefficients (Kd).
        nuclide_water_diffusivity_by_rock (dict[str, NuclidePropertyData]):
            Unit-specific nuclide diffusivities in water.
    """

    nuclide_name: str
    species: SpeciesType | None = None
    emitted_energy: EmittedEnergy | None = None
    sorption_by_rock: dict[str, NuclidePropertyData] = field(
        default_factory=dict
    )
    nuclide_water_diffusivity_by_rock: dict[str, NuclidePropertyData] = field(
            default_factory=dict
        )

    
    def get_sorption_coefficient(self, rock_name):
        """Return the Kd value for this nuclide in one geological unit.

        Args:
            rock_name (str): Geological-unit name.

        Returns:
            NuclidePropertyData: Kd data for the requested unit.

        Raises:
            KeyError: If no sorption data exist for the unit.
        """
        try:
            return self.sorption_by_rock[rock_name]
        except KeyError as error:
            available = ", ".join(self.sorption_by_rock)

            raise KeyError(
                f"No sorption coefficient for '{self.nuclide_name}' "
                f"in unit '{rock_name}'. Available units: {available}"
            ) from error

    def get_nuclide_water_diffusivity(self, rock_name):
            """Return the water-diffusivity value for this nuclide in one
            geological unit.
    
            Args:
                rock_name (str): Geological-unit name.
    
            Returns:
                RockScalarProperty: Nuclide water-diffusivity data for the
                    requested unit.
    
            Raises:
                KeyError: If no diffusivity data exist for the unit.
            """
            try:
                return self.nuclide_water_diffusivity_by_rock[rock_name]
            except KeyError as error:
                available = ", ".join(self.nuclide_water_diffusivity_by_rock)
    
                raise KeyError(
                    f"No water diffusivity for '{self.nuclide_name}' "
                    f"in unit '{rock_name}'. Available units: {available}"
                ) from error


@dataclass
class NuclideData:
    """Store all nuclide data used in a simulation.

    Attributes:
        by_nuclide_name (dict[str, Nuclide]): Mapping from nuclide names to
            their complete global and unit-specific data.
    """

    by_nuclide_name: dict[str, Nuclide] = field(default_factory=dict)

    @classmethod
    def from_yaml_files(
        cls,
        emitted_energy_folder,
        species_type_folder,
        sorption_data_folder,
        nuclide_water_diffusivity_folder,
    ):
        """Load all nuclide data from global and unit-specific YAML files.

        Args:
            emitted_energy_folder (str or pathlib.Path): Path to folder that contains 
                `emitted_energy.yaml`.
            species_type_folder (str or pathlib.Path): Path to folder that contains the 
                species-type YAML file.
            sorption_data_folder (str or pathlib.Path): Folder
                containing one YAML file per geological unit.
            nuclide_water_diffusivity_folder (str or pathlib.Path): Folder containing one YAML file per geological unit.

        Returns:
            NuclideData: Combined nuclide database.
        """
        
        sorption_data_folder = Path(sorption_data_folder)
        nuclide_water_diffusivity_folder = Path(nuclide_water_diffusivity_folder)

        emitted_energy_folder = Path(emitted_energy_folder)
        emitted_energy_file = sorted(emitted_energy_folder.glob("*.yaml"))[0]

        species_data_folder = Path(species_type_folder)
        species_type_file = sorted(species_data_folder.glob("*.yaml"))[0]

        emitted_energy_data = cls._load_yaml(emitted_energy_file)
        species_type_data = cls._load_yaml(species_type_file)

        # Every nuclide name that appears in any of the three dictionaries is included.
        nuclide_names = (
            set(emitted_energy_data)
            | set(species_type_data)
        )

        sorption_by_rock = cls._load_rock_nuclide_data(sorption_data_folder)
        nuclide_water_diffusivity_by_rock = cls._load_rock_nuclide_data(nuclide_water_diffusivity_folder)

        nuclides = {}

        for nuclide_name in sorted(nuclide_names):
            energy_data = emitted_energy_data.get(nuclide_name)
            emitted_energy = None
            if energy_data is not None:
                emitted_energy = cls._create_emitted_energy(energy_data)

            species_nuclide_data = species_type_data.get(nuclide_name)
            species_data = None
            if species_nuclide_data is not None:
                species_data = cls._create_species_type(
                    species_nuclide_data
                )

            nuclides[nuclide_name] = Nuclide(
                            nuclide_name=nuclide_name,
                            species=species_data,
                            emitted_energy=emitted_energy,
                            sorption_by_rock=cls._create_nuclide_properties_by_rock(
                                nuclide_name, sorption_by_rock
                            ),
                            nuclide_water_diffusivity_by_rock=cls._create_nuclide_properties_by_rock(
                                nuclide_name, nuclide_water_diffusivity_by_rock
                            )
                        )

        return cls(by_nuclide_name=nuclides)
    
    @staticmethod
    def _create_species_type(data):
        """Create species metadata from one YAML entry.

        Args:
            data (dict or None): Species-type data for one nuclide.

        Returns:
            SpeciesType or None: Parsed species data, or None if the
            nuclide has no species entry.
        """
        if data is None:
            return None

        return SpeciesType(
            source=data["source"],
            species_type=data["species_type"],
            element_category=data["element_category"],
            description=data["description"],
        )

    @staticmethod
    def _create_emitted_energy(data):
        """Create emitted-energy metadata from one YAML entry.

        Args:
            data (dict or None): Emitted-energy data for one nuclide.

        Returns:
            EmittedEnergy or None: Parsed energy data, or None if the
            nuclide has no energy entry.
        """
        if data is None:
            return None

        return EmittedEnergy(
            source=data["source"],
            alpha=float(data["alpha"]),
            electron=float(data["electron"]),
            photon=float(data["photon"]),
            total=float(data["total"]),
            unit_str=data.get("unit_str"),
            unit_base=tuple(data["unit_base"]),
        )
    
    @classmethod
    def _create_nuclide_properties_by_rock(
        cls,
        nuclide_name,
        rock_nuclide_data,
    ):
        """Create properties for a nuclide across all rock units.

        Args:
            nuclide_name (str): Nuclide whose properties are created.
            rock_nuclide_data (dict[str, dict]): Raw property data
                indexed by geological-unit name.

        Returns:
            dict[str, RockScalarProperty]: Scalar properties indexed
                by geological-unit name.
        """
        nuclide_properties_by_rock = {}

        for rock_name, properties_list in rock_nuclide_data.items():
            nuclide_entries = properties_list.get(nuclide_name)

            if not nuclide_entries:
                continue

            nuclide_property_data = nuclide_entries[0]

            tag = nuclide_property_data.get("tag", {})
            
            distribution = nuclide_property_data[
                "probability_distribution"
            ]

            nuclide_properties_by_rock[rock_name] = NuclidePropertyData(
                                                    nuclide_name=nuclide_name,
                                                    property_type=nuclide_property_data["type"],
                                                    source=nuclide_property_data["source"],
                                                    value=None if nuclide_property_data["value"] is None
                                                    else float(nuclide_property_data["value"]),
                                                    value_min=None if nuclide_property_data["value_min"] is None
                                                    else float(nuclide_property_data["value_min"]),
                                                    value_max=None if nuclide_property_data["value_max"] is None
                                                    else float(nuclide_property_data["value_max"]),
                                                    value_std=None if nuclide_property_data["value_std"] is None
                                                    else float(nuclide_property_data["value_std"]),
                                                    sample_size=int(
                                                        distribution["sample_size"]
                                                    ),
                                                    sampled_data=distribution["sampled_data"],
                                                    unit_str=nuclide_property_data.get("unit_str"),
                                                    unit_base=tuple(nuclide_property_data["unit_base"]),
                                                    description=nuclide_property_data["description"],
                                                    property_id=tag["ID"],
                                                )
              

        return nuclide_properties_by_rock

    @classmethod
    def _load_rock_nuclide_data(cls, folder):
        """Load unit-specific nuclide properties from one folder.

        Args:
            folder (pathlib.Path): Folder containing one YAML file per
                geological unit.

        Returns:
            dict[str, dict]: Raw nuclide-property data indexed by
                geological-unit name.
        """
        rock_nuclide_files = sorted(
            (
                *folder.glob("*.yaml"),
                *folder.glob("*.yml"),
            )
        )

        rock_nuclide_data = {}

        for rock_nuclide_file in rock_nuclide_files:
            rock_name = cls._rock_name_from_file(rock_nuclide_file)
            rock_nuclide_data[rock_name] = cls._load_yaml(
                rock_nuclide_file
            )

        return rock_nuclide_data

    @staticmethod
    def _load_yaml(filename):
        """Load one YAML file safely.

        Args:
            filename (pathlib.Path): YAML-file path.

        Returns:
            dict: Parsed YAML content.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not filename.is_file():
            raise FileNotFoundError(
                f"YAML file was not found: {filename}"
            )

        with filename.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return data

    @staticmethod
    def _rock_name_from_file(filename):
        """Infer a geological-rock name from a rock YAML filename.

        Args:
            filename (pathlib.Path): Unit-property YAML filename.

        Returns:
            str: Filename stem with an optional trailing `-number`
            suffix removed.
        """
        return re.sub(r"-\d+$", "", filename.stem)

    def get(self, nuclide_name):
        """Return data for one nuclide.

        Args:
            nuclide_name (str): Nuclide name, for example "I-129".

        Returns:
            Nuclide: Complete data object for the nuclide.

        Raises:
            KeyError: If the requested nuclide is unavailable.
        """
        try:
            return self.by_nuclide_name[nuclide_name]
        except KeyError as error:
            available = ", ".join(self.by_nuclide_name)

            raise KeyError(
                f"Nuclide '{nuclide_name}' was not found. "
                f"Available nuclides: {available}"
            ) from error

    @property
    def nuclide_names(self):
        """Return all loaded nuclide names.

        Returns:
            list[str, ...]: Loaded nuclide names.
        """
        return list(self.by_nuclide_name)
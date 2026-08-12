from dataclasses import dataclass, field
import math
from typing import Literal, Callable

from yaml2nuctrans.primary_parameters.rock_database import RockData, RockUnit
from yaml2nuctrans.primary_parameters.nuclide_database import NuclideData

PhaseType = Literal["AqueousLiquid", "Solid"]
PropertyType = Literal["Constant", "Function"]

@dataclass(frozen=True)
class NuclideProperty:
    """Store one resolved OGS material-property definition.
    """

    property_type: PropertyType 
    value: float | str


NuclideCalculator = Callable[
    [RockUnit, object, "DiffusionModel"],
    float | str,
]

@dataclass(frozen=True)
class DiffusionModel:
    """Store default parameters and methods for pore diffusion.
    """

    anion_exclusion_coefficient: float = 0.45
    sphere_distribution_parameter: float = 0.5

    def accessible_porosity(
        self,
        porosity,
        species_type,
        simplified_lithology,
    ):
        """Calculate the porosity accessible to a dissolved species.

        Anions receive the anion-exclusion factor only if the unit is
        classified as mudstone. All other species use total porosity.

        Args:
            porosity (float): Total medium porosity.
            species_type (str): Species class, for example "anion".
            simplified_lithology (list[str]): Lithology classes of the
                geological unit.

        Returns:
            float: Accessible porosity.

        Raises:
            ValueError: If porosity is not in the interval (0, 1].
        """
        if not 0.0 < porosity <= 1.0:
            raise ValueError(
                "Porosity must be greater than zero and at most one."
            )

        is_anion = species_type.lower() == "anion"
        is_mudstone = "Mudstone" in simplified_lithology

        if is_anion and is_mudstone:
            return self.anion_exclusion_coefficient * porosity

        return porosity

    def tortuosity(self, accessible_porosity):
        """Calculate the diffusion tortuosity factor.

        The relation is based on randomly distributed spheres:
        `1 - p * log(accessible_porosity)`.

        Args:
            accessible_porosity (float): Porosity accessible to the
                dissolved species.

        Returns:
            float: Dimensionless tortuosity factor.

        Raises:
            ValueError: If accessible porosity is not in (0, 1].
        """
        if not 0.0 < accessible_porosity <= 1.0:
            raise ValueError(
                "Accessible porosity must be greater than zero and "
                "at most one."
            )

        return 1.0 - (
            self.sphere_distribution_parameter
            * math.log(accessible_porosity)
        )

    def pore_diffusion(
        self,
        porosity,
        species_type,
        nuclide_diffusion_in_water,
        simplified_lithology
    ):
        """Calculate the pore diffusion coefficient.

        The method first computes accessible porosity, then tortuosity,
        and finally applies:

        `D_p = D_w / tortuosity`

        Args:
            porosity (float): Total medium porosity.
            species_type (str): Species class, for example "anion".
            nuclide_diffusion_in_water (float): Reference nuclide diffusion coefficient in water.
            simplified_lithology (list[str]): Lithology classes.

        Returns:
            float: Pore diffusion coefficient in m^2/s.
        """
        accessible_porosity = self.accessible_porosity(
            porosity=porosity,
            species_type=species_type,
            simplified_lithology=simplified_lithology,
        )

        tortuosity = self.tortuosity(accessible_porosity)

        return nuclide_diffusion_in_water / tortuosity


def default_retardation_factor(
    rock_unit,
    nuclide,
    diffusion_model=None,
):
    """Calculate equilibrium-sorption retardation.

    Args:
        rock_unit (RockUnit): Rock unit of the current medium.
        nuclide (Nuclide): Nuclide being transported.
        diffusion_model (DiffusionModel): Unused default
            diffusion model.

    Returns:
        float: Dimensionless retardation factor.
    """
    density = rock_unit.get_property(
        "density"
    ).value_for_run

    porosity = rock_unit.get_property(
        "porosity"
    ).value_for_run

    sorption_coefficient = nuclide.get_sorption_coefficient(
        rock_unit.rock_name
    ).value_for_run

    return (
        1.0
        + density
        * sorption_coefficient
        / porosity
    )

def default_decay_rate(
    rock_unit=None,
    nuclide=None,
    diffusion_model=None,
):
    """Return the default decay rate.

    Args:
        rock_unit (RockUnit): Unused rock-unit data.
        nuclide (Nuclide): Unused nuclide data.
        diffusion_model (DiffusionModel): Unused diffusion model.

    Returns:
        float: Default decay rate in 1/s.
    """
    return 0.0

def default_pore_diffusion(
    rock_unit,
    nuclide,
    diffusion_model,
):
    """Calculate pore diffusion from unit and species data.

    Args:
        rock_unit (RockUnit): Rock unit of the current medium.
        nuclide (Nuclide): Nuclide being transported.
        diffusion_model (DiffusionModel): Diffusion model used
            for the calculation.

    Returns:
        float: Pore diffusion coefficient in m^2/s.
    """
    porosity = rock_unit.get_property(
        "porosity"
    ).value_for_run

    nuclide_water_diffusivity = nuclide.get_nuclide_water_diffusivity(
            rock_unit.rock_name
        ).value_for_run

    return diffusion_model.pore_diffusion(
        porosity=porosity,
        species_type=nuclide.species.species_type,
        nuclide_diffusion_in_water=nuclide_water_diffusivity,
        simplified_lithology=rock_unit.simplified_lithology,
    )


@dataclass(frozen=True)
class NuclideProperties:
    """Store resolved OGS component properties for one unit and nuclide.

    Attributes:
        rock_name (str): rock name.
        phase_type (PhaseType) : OGS phase containing the nuclide.
        nuclide_name (str): Nuclide name, for example "I-129".
        properties (dict[str, NuclideProperty]): Resolved OGS
            component properties such as pore diffusion, retardation
            factor, and decay rate.
    """

    rock_name: str
    phase_type: PhaseType
    nuclide_name: str
    properties: dict[str, NuclideProperty] = field(
        default_factory=dict
    )

    @property
    def pore_diffusion(self):
        """Return the pore-diffusion property.

        Returns:
            NuclideProperty: Resolved pore-diffusion definition.
        """
        return self.properties["pore_diffusion"]

    @property
    def retardation_factor(self):
        """Return the retardation-factor property.

        Returns:
            NuclideProperty: Resolved retardation-factor definition.
        """
        return self.properties["retardation_factor"]

    @property
    def decay_rate(self):
        """Return the decay-rate property.

        Returns:
            NuclideProperty: Resolved decay-rate definition.
        """
        return self.properties["decay_rate"]

    @classmethod
    def from_databases(
        cls,
        rock_data,
        nuclide_data,
        rock_name,
        phase_type,
        nuclide_name,
        calculators=None,
        diffusion_model=None,
    ):
        """Create component properties from rock and nuclide databases.

        User calculators replace default calculators with the same OGS
        property name. Each calculator receives `rock_unit`, `nuclide`,
        and `diffusion_model`, and must return a numerical value.

        Args:
            rock_data (RockData): Database of rock properties.
            nuclide_data (NuclideData): Database of nuclide
                species, energy, and sorption data.
            rock_name (str): Geological unit for the calculation.
            phase_type (PhaseType): OGS phase type.
            nuclide_name (str): Nuclide for the calculation.
            calculators (dict[str, callable] or None): Optional custom
                property calculators keyed by OGS property name.
            diffusion_model (DiffusionModel or None): Diffusion model
                and parameters. Defaults to `DiffusionModel()`.

        Returns:
            NuclideProperties: Resolved OGS component properties.

        """

        rock_unit = rock_data.get(rock_name)
        nuclide = nuclide_data.get(nuclide_name)
        diffusion_model = diffusion_model or DiffusionModel()

        default_calculators = cls._default_calculators(
            phase_type=phase_type
        )

        active_calculators = {
            **default_calculators,
            **(calculators or {}),
        }

        properties = {}

        for property_name, calculator in active_calculators.items():
            property_value = calculator(rock_unit, nuclide, diffusion_model)
            property_type = "Constant"

            if isinstance(property_value, str):
                property_type = "Function"

            properties[property_name] = NuclideProperty(
                property_type=property_type,
                value=property_value)

        return cls(
            rock_name=rock_name,
            phase_type=phase_type,
            nuclide_name=nuclide_name,
            properties=properties,
        )

    @staticmethod
    def _default_calculators(
        phase_type
    ):
        """Return default calculators for the requested phase.

        Args:
            phase_type (PhaseType): AqueousLiquid or Solid.

        Raises:
            ValueError: If the phase type is not AqueousLiquid or Solid.

        Returns:
            dict[str, callable]: Default property calculators for the specified phase type.
        """
        
        if phase_type == "AqueousLiquid":
            return {
                "pore_diffusion": default_pore_diffusion,
                "retardation_factor": default_retardation_factor,
                "decay_rate": default_decay_rate,
            }

        if phase_type == "Solid":
            return {}

        raise ValueError(
            f"Unsupported phase type: {phase_type}"
        )

@dataclass(frozen=True)
class NuclidePropertiesCollection:
    """Store resolved component properties for all units and nuclides.

    Attributes:
       by_rock: Resolved component properties arranged as:

        ``rock_name -> phase_type -> nuclide_name -> properties``.
    """

    by_rock: dict[
        str,
        dict[PhaseType, dict[str, NuclideProperties]],
    ] = field(default_factory=dict)

    @classmethod
    def from_databases(
        cls,
        rock_data,
        nuclide_data,
        calculators_by_rock_name=None,
        diffusion_model=None,
    ):
        """Create resolved component properties for many units and nuclides.

        Args:
            rock_data (RockData): Database of rock properties.
            nuclide_data (NuclideData): Database of nuclide data.
            calculators_by_rock_name (dict[str, dict[callable, dict[str, callable]]] or None):
                Optional custom calculators for individual units.
                 ``rock -> phase -> OGS property -> calculator``
            diffusion_model (DiffusionModel or None): Diffusion model
                shared by all calculations.

        Returns:
            NuclidePropertiesCollection: All resolved unit/nuclide
            component-property definitions.
        """
        
        calculators_by_rock_name = calculators_by_rock_name or {}
        rock_names = rock_data.rock_names
        nuclide_names = nuclide_data.nuclide_names

        by_rock = {}
        for rock_name in rock_names:
            calculators_by_phase = (calculators_by_rock_name.get(rock_name, {}))
            
            phase_types: list[PhaseType] = [
                "AqueousLiquid",
            ]

            for phase_type in calculators_by_phase:
                if phase_type not in phase_types:
                    phase_types.append(phase_type)

            by_rock[rock_name] = {}

            for phase_type in phase_types:
                by_rock[rock_name][phase_type] = {}

                calculators = calculators_by_phase.get(
                    phase_type
                )
                for nuclide_name in nuclide_names:
                    
                    by_rock[rock_name][phase_type][nuclide_name] = (
                        NuclideProperties.from_databases(
                            rock_data=rock_data,
                            nuclide_data=nuclide_data,
                            rock_name=rock_name,
                            phase_type=phase_type,
                            nuclide_name=nuclide_name,
                            calculators=calculators,
                            diffusion_model=diffusion_model,
                        )
                    )

        return cls(by_rock=by_rock)

    def get(self, rock_name, phase_type, nuclide_name):
        """Return component properties for one unit and nuclide.

        Args:
            rock_name (str): rock name.
            phase_type (PhaseType): AqueousLiquid or Solid.
            nuclide_name (str): Nuclide name.

        Returns:
            NuclideProperties: Resolved component properties.
        """
        return self.by_rock[rock_name][phase_type][nuclide_name]
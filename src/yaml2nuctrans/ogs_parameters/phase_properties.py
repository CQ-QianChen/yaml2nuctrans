from dataclasses import dataclass, field
from typing import  Callable, Literal

from yaml2nuctrans.primary_parameters.rock_database import RockData, RockUnit

PhaseType = Literal["AqueousLiquid", "Solid"]
PropertyType = Literal["Constant", "Function"]

@dataclass
class PhaseProperty:
    """Store one OGS material property.

    Attributes:
        type: OGS property type.
        value: Constant value or OGS function expression.
    """

    property_type: PropertyType
    value: float | str

PhaseCalculator = Callable[[RockUnit], float | str]

@dataclass
class Phase:
    """Store properties belonging to one material phase.

    Attributes:
        properties: Mapping from property name to its definition.
    """

    properties: dict[str, PhaseProperty] = field(default_factory=dict)

    @property
    def density(self):
        """Return the density property.

        Returns:
            PhaseProperty: The density property.
        """
        return self.properties["density"]

    @property
    def viscosity(self):
        """Return the dynamic-viscosity property.

        Returns:
            PhaseProperty: The dynamic-viscosity property.
        """
        return self.properties["viscosity"]

    @classmethod
    def from_rock_data(
        cls,
        rock_data,
        rock_name,
        phase_type,
        calculators= None,
    ):
        """Build properties for one phase of one rock unit.

        For ``AqueousLiquid``, density and viscosity default to
        1000.0 kg/m³ and 0.001 Pa·s, respectively.

        For ``Solid``, no properties are added automatically. Supply
        calculators when a solid phase is needed.

        User-provided calculators overwrite default calculators with
        the same property name and may define further properties.

        Args:
            rock_data (RockData): Database containing loaded rock-unit data.
            rock_name (str): Geological unit name.
            phase_type (PhaseType): OGS phase type.
            calculators (dict[str, PhaseCalculator]): Mapping from OGS property names to functions.
                Each function receives a RockUnit and returns either a
                float for a Constant property or a string for a
                Function property.

        Returns:
            Phase: Resolved OGS phase properties.
        """
        rock_unit = rock_data.get(rock_name)

        default_calculators = cls._default_calculators(
            phase_type=phase_type,
        )

        active_calculators = {
            **default_calculators,
            **(calculators or {}),
        }

        properties = {}

        for property_name, calculator in (
            active_calculators.items()
        ):
            property_value = calculator(rock_unit)

            property_type: PropertyType = "Constant"

            if isinstance(property_value, str):
                property_type = "Function"

            properties[property_name] = PhaseProperty(
                property_type=property_type,
                value=property_value,
            )

        return cls(properties=properties)
    
    @staticmethod
    def _default_calculators(phase_type):
        """Return default property calculators for a phase type.

        Args:
            phase_type (PhaseType): AqueousLiquid or Solid.

        Raises:
            ValueError: If the phase type is not AqueousLiquid or Solid.

        Returns:
            dict[str, PhaseCalculator]: Default property calculators for the specified phase type.
        """
        if phase_type == "AqueousLiquid":
            return {
                "density": lambda rock: 1000.0,
                "viscosity": lambda rock: 0.001,
            }

        if phase_type == "Solid":
            return {}

        raise ValueError(
            f"Unsupported phase type: {phase_type}"
        )


@dataclass
class PhaseProperties:
    """Store phase properties for all geological rock units.

    Attributes:
        by_unit: Mapping structured as:

            ``rock name -> phase type -> Phase``

        For example:

            ``{"Host_rock": {"AqueousLiquid": Phase(...)}}``
    """

    by_rock_name: dict[str, dict[PhaseType, Phase]] = field(
        default_factory=dict
    )

    @classmethod
    def from_rock_data(
        cls,
        rock_data,
        calculators_by_rock_name= None,
    ):
        """Create phase properties for all rock units.

        Every rock unit receives an ``AqueousLiquid`` phase with
        default density and viscosity.

        A phase appears when it is either:

        - ``AqueousLiquid`` (always created), or
        - present in ``calculators_by_rock_name`` for that rock.

        Args:
            rock_data (RockData): Database containing all loaded rock units.
            calculators_by_rock_name (dict[str,dict[PhaseType, dict[str, PhaseCalculator]]]): Optional nested calculators:

                ``{rock_name: {phase_type: {property: calculator}}}``

        Returns:
            PhaseProperties: Resolved phases indexed by rock name.

        """
        calculators_by_rock_name = (
            calculators_by_rock_name or {}
        )

        phases_by_rock_name = {}

        for rock_name in rock_data.rock_names:
            calculators_by_phase = (
                calculators_by_rock_name.get(rock_name, {})
            )

            phase_types: list[PhaseType] = [
                "AqueousLiquid",
            ]

            for phase_type in calculators_by_phase:
                if phase_type not in phase_types:
                    phase_types.append(phase_type)

            phases = {}

            for phase_type in phase_types:
                phases[phase_type] = Phase.from_rock_data(
                    rock_data=rock_data,
                    rock_name=rock_name,
                    phase_type=phase_type,
                    calculators=calculators_by_phase.get(
                        phase_type
                    ),
                )

            phases_by_rock_name[rock_name] = phases

        return cls(by_rock_name=phases_by_rock_name)

    def get(self, rock_name, phase_type):
        """Return one resolved phase for a geological rock unit.

        Args:
            rock_name (str): Geological unit name.
            phase_type (PhaseType): Requested OGS phase type.

        Returns:
            Phase: Resolved phase properties.

        Raises:
            KeyError: If the rock unit or phase does not exist.
        """
        try:
            return self.by_rock_name[rock_name][phase_type]
        except KeyError as error:
            available_rocks = ", ".join(self.by_rock_name)

            raise KeyError(
                f"Phase '{phase_type}' for rock '{rock_name}' "
                "was not found. "
                f"Available rocks: {available_rocks}"
            ) from error

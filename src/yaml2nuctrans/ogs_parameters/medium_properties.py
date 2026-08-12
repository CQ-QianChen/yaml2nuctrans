from dataclasses import dataclass, field
from typing import Literal, Callable
from yaml2nuctrans.primary_parameters.rock_database import RockData, RockUnit


@dataclass(frozen=True)
class MediumProperty:
    property_type: Literal["Constant", "Function"]
    value: float | str

# A calculator receives one RockUnit and returns a numerical OGS value.
MediumCalculator = Callable[[RockUnit], float|str]

@dataclass(frozen=True)
class Medium:
    """
    OGS medium properties for one geological unit. The properties dictionary can contain any additional medium
    property, not only the four properties.
    """

    # If no value is provided, create a new empty dictionary.
    properties: dict[str, MediumProperty] = field(default_factory=dict) 

    @property
    def longitudinal_dispersivity(self):
        """Return the longitudinal dispersivity property

        Returns:
            MediumProperty: The resolved longitudinal-dispersivity property.
        """
        return self.properties["longitudinal_dispersivity"]

    @property
    def transversal_dispersivity(self):
        """Return the transversal dispersivity property.

        Returns:
            MediumProperty: The resolved transversal-dispersivity property.
        """
        return self.properties["transversal_dispersivity"]

    @property
    def permeability(self):
        """Return the intrinsic permeability property.

        Returns:
            MediumProperty: The resolved permeability property.
        """
        return self.properties["permeability"]

    @property
    def porosity(self):
        """Return the porosity property.

        Returns:
            MediumProperty: The resolved porosity property.
        """
        return self.properties["porosity"]

    @classmethod
    def from_rock_data(cls, rock_data, rock_name, calculators= None):
        """Build medium properties from one RockUnit in a RockDatabase.

        User-provided calculators replace the default calculation for
        the same property name.

        Args:
            rock_data (RockData): Database containing loaded rock-unit data.
            rock_name (str): Name of the rock unit.
            calculators (dict[str, callable], optional): Optional mapping
                from OGS property names to custom functions. Each
                function receives one RockUnit and returns a value. Defaults to None.

        Returns:
            Medium: Medium with resolved property values.
        """
        
        rock_unit = rock_data.get(rock_name)

        default_calculators: dict[str, MediumCalculator] = {
            "longitudinal_dispersivity": (
                lambda rock: 0.0
            ),
            "transversal_dispersivity": (
                lambda rock: 0.0
            ),
            "permeability": (
                lambda rock: rock.get_property(
                    "intrinsic_permeability"
                ).value_for_run
            ),
            "porosity": (
                lambda rock: rock.get_property(
                    "porosity"
                ).value_for_run
            ),
        }


        # creates a new dictionary by merging two dictionaries, with the second one overriding the first if keys overlap.
        active_calculators = {
            **default_calculators,
            **(calculators or {}),
        }

        properties = {}
        for property_name, calculator in active_calculators.items():
            property_value = calculator(rock_unit)
            property_type = "Constant"
            if isinstance(property_value, str):
                property_type = "Function"
        
            properties[property_name] = MediumProperty(
                    property_type=property_type,
                    value=property_value)

        return cls(properties=properties)



@dataclass(frozen=True)
class MediumProperties:
    """Maps geological-unit names to their medium properties."""
    by_rock_name: dict[str, Medium] = field(default_factory=dict)

    @classmethod
    def from_rock_data(
        cls,
        rock_data,
        calculators_by_rock_name=None,
    ):
        """Create all OGS media from rock data.

        A custom calculator replaces the default
        calculator with the same property name.

        Args:
            rock_data (RockData): Database containing all loaded rock-unit data.
            calculators_by_rock_name (dict[str, dict[str, callable]] or None):
                Optional nested mapping in the form
                `{rock_name: {ogs_property_name: calculator}}`.

        Returns:
            MediumProperties: Resolved OGS media mapped by unit name.
        """
        calculators_by_rock_name = calculators_by_rock_name or {}

        media = {}

        for rock_name in rock_data.rock_names:
            media[rock_name] = Medium.from_rock_data(
                rock_data=rock_data,
                rock_name=rock_name,
                calculators=calculators_by_rock_name.get(rock_name),
            )

        return cls(by_rock_name=media)

    def get(self, rock_name):
        """Return resolved medium properties for one geological unit.

        Args:
            rock_name (str): Name of the geological unit.

        Returns:
            Medium: Resolved OGS medium properties.

        Raises:
            KeyError: If the unit does not exist in this container.
        """
        try:
            return self.by_rock_name[rock_name]
        except KeyError as error:
            available = ", ".join(self.by_rock_name)
            raise KeyError(
                f"Medium '{rock_name}' was not found. "
                f"Available units: {available}"
            ) from error
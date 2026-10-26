import pytest

from yaml2nuctrans.config.ogs_sim_config import PointSource1D
from yaml2nuctrans.primary_parameters.geometry import Geometry, GeometryData, Units1D
from yaml2nuctrans.primary_parameters.nuclide_database import (
    EmittedEnergy,
    Nuclide,
    NuclideData,
    NuclidePropertyData,
    SpeciesType,
)
from yaml2nuctrans.primary_parameters.rock_database import (
    RockData,
    RockProperty,
    RockUnit,
)

# Shared fixtures for a manually built, no-uncertainty Claystone, Cm-247 decay-chain example.

ROCK_NAME = "Claystone"
TOP_ELEVATION = 0.0
THICKNESS = 10.0
NUCLIDE_NAMES = ["Cm-247", "Am-243", "Pu-239", "U-235", "Pa-231", "Ac-227"]
DIFFUSIVITY_VALUE = 7.5e-10  # m^2/s
SORPTION_VALUE = 0.5  # m^3/kg


@pytest.fixture
def rock_name():
    return ROCK_NAME


@pytest.fixture
def nuclide_names():
    return list(NUCLIDE_NAMES)


def _make_rock_property(property_name, value, unit_str, unit_base, property_id):
    return RockProperty(
        property_name=property_name,
        property_type="scalar",
        source="manual",
        value=value,
        value_min=value,
        value_max=value,
        value_std=0.0,
        sample_size=1,
        sampled_data="",
        unit_str=unit_str,
        unit_base=unit_base,
        description=f"Manually set {property_name} for Claystone, no uncertainty.",
        property_id=property_id,
    )


@pytest.fixture
def rock_data():
    """A RockData with a single Claystone unit, no uncertainty."""
    claystone = RockUnit(
        rock_name=ROCK_NAME,
        simplified_lithology=["Claystone"],
        properties={
            "porosity": _make_rock_property(
                "porosity", 0.137, None, (0, 0, 0, 0, 0, 0, 0), "claystone-porosity"
            ),
            "density": _make_rock_property(
                "density",
                2330.0,
                "kg/m^3",
                (1, -3, 0, 0, 0, 0, 0),
                "claystone-density",
            ),
            "intrinsic_permeability": _make_rock_property(
                "intrinsic_permeability",
                2.0e-20,
                "m^2",
                (0, 2, 0, 0, 0, 0, 0),
                "claystone-intrinsic-permeability",
            ),
        },
    )
    return RockData(by_rock_name={ROCK_NAME: claystone})


def _make_nuclide_property(
    nuclide_name, property_name, value, unit_str, unit_base, property_id
):
    return NuclidePropertyData(
        nuclide_name=nuclide_name,
        property_type="scalar",
        source="manual",
        value=value,
        value_min=value,
        value_max=value,
        value_std=0.0,
        sample_size=1,
        sampled_data="",
        unit_str=unit_str,
        unit_base=unit_base,
        description=f"Manually set {property_name} for {nuclide_name}, no uncertainty.",
        property_id=property_id,
    )


@pytest.fixture
def nuclide_data():
    """NuclideData for the Cm-247 decay chain, all properties uniform
    and certain, in the single Claystone unit."""
    nuclides = {}

    for nuclide_name in NUCLIDE_NAMES:
        slug = nuclide_name.lower().replace("-", "")

        diffusivity = _make_nuclide_property(
            nuclide_name,
            "nuclide_water_diffusivity",
            DIFFUSIVITY_VALUE,
            "m^2/s",
            (0, 2, -1, 0, 0, 0, 0),
            f"{slug}-diffusivity",
        )
        sorption = _make_nuclide_property(
            nuclide_name,
            "sorption",
            SORPTION_VALUE,
            "m^3/kg",
            (-1, 3, 0, 0, 0, 0, 0),
            f"{slug}-sorption",
        )
        species = SpeciesType(
            source="manual",
            species_type="cation",
            element_category="actinide",
            description=f"{nuclide_name}: manually set for testing.",
        )
        emitted_energy = EmittedEnergy(
            source="manual",
            alpha=0.0,
            electron=0.0,
            photon=0.0,
            total=1.0e-13,
            unit_str="kg*m^3/s^2",
            unit_base=(1, 2, -2, 0, 0, 0, 0),
        )

        nuclides[nuclide_name] = Nuclide(
            nuclide_name=nuclide_name,
            species=species,
            emitted_energy=emitted_energy,
            sorption_by_rock={ROCK_NAME: sorption},
            nuclide_water_diffusivity_by_rock={ROCK_NAME: diffusivity},
        )

    return NuclideData(by_nuclide_name=nuclides)


def _make_geometry_interface(interface_name, value, property_id):
    return Geometry(
        interface_name=interface_name,
        property_type="scalar",
        source="manual",
        value=value,
        value_min=value,
        value_max=value,
        value_std=0.0,
        sample_size=1,
        sampled_data="",
        unit_str="m",
        unit_base=(0, 1, 0, 0, 0, 0, 0),
        description=f"Manually set {interface_name}, no uncertainty.",
        property_id=property_id,
    )


@pytest.fixture
def geometry_data():
    """GeometryData for a single 10 m thick Claystone unit."""
    return GeometryData(
        by_rock_interface={
            f"{ROCK_NAME}_top": _make_geometry_interface(
                f"{ROCK_NAME}_top", TOP_ELEVATION, "claystone-top"
            ),
            f"{ROCK_NAME}_bottom": _make_geometry_interface(
                f"{ROCK_NAME}_bottom", TOP_ELEVATION - THICKNESS, "claystone-bottom"
            ),
        }
    )


@pytest.fixture
def units_1d(geometry_data):
    """Units1D for a single 10 m thick Claystone unit."""
    return Units1D.from_geometry_data(geometry_data)


@pytest.fixture
def source():
    """A point source at the top of the Claystone unit."""
    return PointSource1D(source_type="point", loc=TOP_ELEVATION)

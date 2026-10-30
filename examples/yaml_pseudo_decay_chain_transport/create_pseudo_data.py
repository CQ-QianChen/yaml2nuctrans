from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"

# The Cm-247 decay chain
NUCLIDE_NAMES = ["Cm-247", "Am-243", "Pu-239", "U-235", "Pa-231", "Ac-227"]

TOP_ELEVATION = 0.0  # m, top of the shallowest unit

# Rock units, listed top to bottom.
ROCK_UNITS = [
    {
        "name": "Pseudo_Sandstone",
        "lithology": ["Mudstone"],
        "thickness": 10.0,
        "bottom_uncertainty": 2.0,
        "porosity": {
            "value": 0.137,
            "value_min": 0.10,
            "value_max": 0.17,
            "value_std": 0.02,
            "sampled_data": "scipy.stats.uniform(loc=0.10, scale=0.07).rvs(size=1000000, random_state=21)",
        },
        "density": {
            "value": 2330.0,
            "value_min": 2030.0,
            "value_max": 2630.0,
            "value_std": 100.0,
            "sampled_data": "scipy.stats.norm(loc=2330.0, scale=100.0).rvs(size=1000000, random_state=21)",
        },
        "intrinsic_permeability": {
            "value": 2.0e-20,
            "value_min": 1.0e-20,
            "value_max": 4.0e-20,
            "value_std": 6.0e-21,
            "sampled_data": "scipy.stats.lognorm(s=0.3, scale=2.0e-20, loc=0.0).rvs(size=1000000, random_state=21)",
        },
        "sorption": {
            "value": 0.00181,
            "value_min": 0.0,
            "value_max": 0.0158,
            "value_std": 0.00361,
            "sampled_data": "scipy.stats.lognorm(s=1.27, scale=0.000811, loc=0.0).rvs(size=1000000, random_state=21)",
        },
        "diffusivity": {
            "value": 1.99e-09,
            "value_min": 1.13e-09,
            "value_max": 3.39e-09,
            "value_std": 2.4e-10,
            "sampled_data": "scipy.stats.lognorm(s=0.12, scale=1.98e-09, loc=0.0).rvs(size=1000000, random_state=21)",
        },
    },
    {
        "name": "Pseudo_Claystone",
        "lithology": ["Sandstone"],
        "thickness": 15.0,
        "bottom_uncertainty": 3.0,
        "porosity": {
            "value": 0.22,
            "value_min": 0.15,
            "value_max": 0.30,
            "value_std": 0.04,
            "sampled_data": "scipy.stats.uniform(loc=0.15, scale=0.15).rvs(size=1000000, random_state=21)",
        },
        "density": {
            "value": 2200.0,
            "value_min": 1960.0,
            "value_max": 2440.0,
            "value_std": 80.0,
            "sampled_data": "scipy.stats.norm(loc=2200.0, scale=80.0).rvs(size=1000000, random_state=21)",
        },
        "intrinsic_permeability": {
            "value": 5.0e-13,
            "value_min": 2.5e-13,
            "value_max": 1.0e-12,
            "value_std": 1.5e-13,
            "sampled_data": "scipy.stats.lognorm(s=0.3, scale=5.0e-13, loc=0.0).rvs(size=1000000, random_state=21)",
        },
        "sorption": {
            "value": 0.0003,
            "value_min": 0.0,
            "value_max": 0.0022,
            "value_std": 0.0005,
            "sampled_data": "scipy.stats.lognorm(s=1.0, scale=0.00025, loc=0.0).rvs(size=1000000, random_state=21)",
        },
        "diffusivity": {
            "value": 2.5e-09,
            "value_min": 1.4e-09,
            "value_max": 4.3e-09,
            "value_std": 3.0e-10,
            "sampled_data": "scipy.stats.lognorm(s=0.12, scale=2.48e-09, loc=0.0).rvs(size=1000000, random_state=21)",
        },
    },
    {
        "name": "Pseudo_Limestone",
        "lithology": ["Limestone"],
        "thickness": 20.0,
        "bottom_uncertainty": 2.5,
        "porosity": {
            "value": 0.10,
            "value_min": 0.05,
            "value_max": 0.15,
            "value_std": 0.03,
            "sampled_data": "scipy.stats.uniform(loc=0.05, scale=0.10).rvs(size=1000000, random_state=21)",
        },
        "density": {
            "value": 2550.0,
            "value_min": 2310.0,
            "value_max": 2790.0,
            "value_std": 80.0,
            "sampled_data": "scipy.stats.norm(loc=2550.0, scale=80.0).rvs(size=1000000, random_state=21)",
        },
        "intrinsic_permeability": {
            "value": 1.0e-16,
            "value_min": 5.0e-17,
            "value_max": 2.0e-16,
            "value_std": 3.0e-17,
            "sampled_data": "scipy.stats.lognorm(s=0.4, scale=1.0e-16, loc=0.0).rvs(size=1000000, random_state=21)",
        },
        "sorption": {
            "value": 0.0004,
            "value_min": 0.0,
            "value_max": 0.0028,
            "value_std": 0.0006,
            "sampled_data": "scipy.stats.lognorm(s=1.0, scale=0.00033, loc=0.0).rvs(size=1000000, random_state=21)",
        },
        "diffusivity": {
            "value": 1.5e-09,
            "value_min": 0.85e-09,
            "value_max": 2.55e-09,
            "value_std": 1.8e-10,
            "sampled_data": "scipy.stats.lognorm(s=0.12, scale=1.49e-09, loc=0.0).rvs(size=1000000, random_state=21)",
        },
    },
]


def _scalar_entry(
    spec, unit_str, unit_base, description, tag_id, sample_size=1_000_000
):
    """Build a `type: scalar` data entry from a short property spec."""
    return {
        "type": "scalar",
        "source": "pseudo",
        "value": spec["value"],
        "value_min": spec["value_min"],
        "value_max": spec["value_max"],
        "value_std": spec["value_std"],
        "probability_distribution": {
            "sample_size": sample_size,
            "sampled_data": spec["sampled_data"],
        },
        "unit_str": unit_str,
        "unit_base": list(unit_base),
        "description": description,
        "tag": {"ID": tag_id},
    }


def write_rock_data():
    """Write one `rock_data/{ROCK_NAME}.yaml` per rock unit."""
    rock_data_dir = DATA_DIR / "rock_data"
    rock_data_dir.mkdir(parents=True, exist_ok=True)

    for unit in ROCK_UNITS:
        name = unit["name"]
        rock_properties = {
            "porosity": [
                _scalar_entry(
                    unit["porosity"],
                    unit_str=None,
                    unit_base=(0, 0, 0, 0, 0, 0, 0),
                    description=f"Pseudo porosity for the {name} unit.",
                    tag_id=f"pseudo-{name.lower()}-porosity",
                )
            ],
            "density": [
                _scalar_entry(
                    unit["density"],
                    unit_str="kg/m^3",
                    unit_base=(1, -3, 0, 0, 0, 0, 0),
                    description=f"Pseudo bulk density for the {name} unit.",
                    tag_id=f"pseudo-{name.lower()}-density",
                )
            ],
            "intrinsic_permeability": [
                _scalar_entry(
                    unit["intrinsic_permeability"],
                    unit_str="m^2",
                    unit_base=(0, 2, 0, 0, 0, 0, 0),
                    description=f"Pseudo intrinsic permeability for the {name} unit.",
                    tag_id=f"pseudo-{name.lower()}-permeability",
                )
            ],
        }

        with (rock_data_dir / f"{name}.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(rock_properties, file, sort_keys=False)


def write_site_data():
    """Write `site_data/pseudo_site.yaml` listing all rock units."""
    site_data_dir = DATA_DIR / "site_data"
    site_data_dir.mkdir(parents=True, exist_ok=True)

    site_data = {
        "name": "Pseudo_Site",
        "description": "A made-up three-unit site used to showcase the "
        "YAML/folder-driven yaml2nuctrans workflow.",
    }
    for unit in ROCK_UNITS:
        site_data[unit["name"]] = {
            "source": "pseudo",
            "simplified_lithology": unit["lithology"],
        }

    with (site_data_dir / "pseudo_site.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(site_data, file, sort_keys=False)


def write_geometry_data():
    """Write `geometry/rock_interface.yaml`"""
    geometry_dir = DATA_DIR / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    geometry_data = {
        f"{ROCK_UNITS[0]['name']}_top": [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": TOP_ELEVATION,
                "value_min": TOP_ELEVATION,
                "value_max": TOP_ELEVATION,
                "value_std": 0.0,
                "probability_distribution": {
                    "sample_size": 1,
                    "sampled_data": "",
                },
                "unit_str": "m",
                "unit_base": [0, 1, 0, 0, 0, 0, 0],
                "description": f"Top of the {ROCK_UNITS[0]['name']} unit, "
                "fixed at the point source location.",
                "tag": {"ID": f"pseudo-{ROCK_UNITS[0]['name'].lower()}-top"},
            }
        ]
    }

    top = TOP_ELEVATION
    for unit in ROCK_UNITS:
        name = unit["name"]
        bottom = top - unit["thickness"]
        uncertainty = unit["bottom_uncertainty"]
        value_min = bottom - uncertainty
        value_max = bottom + uncertainty
        scale = value_max - value_min
        value_std = scale / (12**0.5)

        geometry_data[f"{name}_bottom"] = [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": bottom,
                "value_min": value_min,
                "value_max": value_max,
                "value_std": value_std,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": f"scipy.stats.uniform(loc={value_min}, "
                    f"scale={scale}).rvs(size=1000000, random_state=21)",
                },
                "unit_str": "m",
                "unit_base": [0, 1, 0, 0, 0, 0, 0],
                "description": f"Bottom of the {name} unit, uniformly "
                "distributed for the ensemble run.",
                "tag": {"ID": f"pseudo-{name.lower()}-bottom"},
            }
        ]
        top = bottom

    with (geometry_dir / "rock_interface.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(geometry_data, file, sort_keys=False)


def write_nuclide_data():
    """Write the four nuclide-data folders for the Cm-247 decay chain"""
    emitted_energy_dir = DATA_DIR / "nuclide_emitted_energy_data"
    species_type_dir = DATA_DIR / "nuclide_species_data"
    sorption_dir = DATA_DIR / "nuclide_sorption_data"
    diffusivity_dir = DATA_DIR / "nuclide_water_diffusivity_data"

    for directory in (
        emitted_energy_dir,
        species_type_dir,
        sorption_dir,
        diffusivity_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    emitted_energy_data = {
        nuclide_name: {
            "source": "ICRP2008",
            "alpha": 5.0e-13,
            "electron": 0.0,
            "photon": 0.0,
            "total": 5.0e-13,
            "unit_str": "kg*m^3/s^2",
            "unit_base": [1, 2, -2, 0, 0, 0, 0],
        }
        for nuclide_name in NUCLIDE_NAMES
    }
    with (emitted_energy_dir / "emitted_energy.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(emitted_energy_data, file, sort_keys=False)

    species_type_data = {
        nuclide_name: {
            "source": "manual",
            "species_type": "cation",
            "element_category": "actinide",
            "description": f"{nuclide_name} is a member of the Cm-247 "
            "actinide decay chain, assumed to be a simple cation here.",
        }
        for nuclide_name in NUCLIDE_NAMES
    }
    with (species_type_dir / "species_type.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(species_type_data, file, sort_keys=False)

    for unit in ROCK_UNITS:
        name = unit["name"]

        sorption_data = {
            nuclide_name: [
                _scalar_entry(
                    unit["sorption"],
                    unit_str="m^3/kg",
                    unit_base=[-1, 3, 0, 0, 0, 0, 0],
                    description=f"Pseudo sorption coefficient (Kd) for "
                    f"{nuclide_name} in the {name} unit, lognormally "
                    "distributed for the ensemble run.",
                    tag_id=f"pseudo-{nuclide_name.lower()}-{name.lower()}-sorption",
                )
            ]
            for nuclide_name in NUCLIDE_NAMES
        }
        with (sorption_dir / f"{name}.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(sorption_data, file, sort_keys=False)

        diffusivity_data = {
            nuclide_name: [
                _scalar_entry(
                    unit["diffusivity"],
                    unit_str="m^2/s",
                    unit_base=[0, 2, -1, 0, 0, 0, 0],
                    description=f"Pseudo water diffusivity for "
                    f"{nuclide_name} in the {name} unit, lognormally "
                    "distributed for the ensemble run.",
                    tag_id=f"pseudo-{nuclide_name.lower()}-{name.lower()}-diffusivity",
                )
            ]
            for nuclide_name in NUCLIDE_NAMES
        }
        with (diffusivity_dir / f"{name}.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(diffusivity_data, file, sort_keys=False)


def main():
    write_rock_data()
    write_site_data()
    write_geometry_data()
    write_nuclide_data()
    print(f"Pseudo data written to {DATA_DIR}")


if __name__ == "__main__":
    main()

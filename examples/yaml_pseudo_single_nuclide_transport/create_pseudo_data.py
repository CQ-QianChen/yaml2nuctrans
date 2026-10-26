from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"

ROCK_NAME = "Pseudo_Claystone"
NUCLIDE_NAME = "I-129"

TOP_ELEVATION = 0.0  # m
THICKNESS = 10.0  # m


def write_rock_data():
    """Write `rock_data/Pseudo_Claystone.yaml`"""
    rock_data_dir = DATA_DIR / "rock_data"
    rock_data_dir.mkdir(parents=True, exist_ok=True)

    rock_properties = {
        "porosity": [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": 0.137,
                "value_min": 0.10,
                "value_max": 0.17,
                "value_std": 0.02,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": "scipy.stats.uniform(loc=0.10, scale=0.07).rvs(size=1000000, random_state=21)",
                },
                "unit_str": None,
                "unit_base": list((0, 0, 0, 0, 0, 0, 0)),
                "description": "Pseudo porosity for a generic claystone unit, "
                "uniformly distributed for the ensemble run.",
                "tag": {"ID": "pseudo-claystone-porosity"},
            }
        ],
        "density": [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": 2330.0,
                "value_min": 2030.0,
                "value_max": 2630.0,
                "value_std": 100.0,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": "scipy.stats.norm(loc=2330.0, scale=100.0).rvs(size=1000000, random_state=21)",
                },
                "unit_str": "kg/m^3",
                "unit_base": list((1, -3, 0, 0, 0, 0, 0)),
                "description": "Pseudo bulk density for a generic claystone unit, "
                "normally distributed for the ensemble run.",
                "tag": {"ID": "pseudo-claystone-density"},
            }
        ],
        "intrinsic_permeability": [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": 2.0e-20,
                "value_min": 1.0e-20,
                "value_max": 4.0e-20,
                "value_std": 6.0e-21,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": "scipy.stats.lognorm(s=0.3, scale=2.0e-20, loc=0.0).rvs(size=1000000, random_state=21)",
                },
                "unit_str": "m^2",
                "unit_base": list((0, 2, 0, 0, 0, 0, 0)),
                "description": "Pseudo intrinsic permeability for a generic "
                "claystone unit, lognormally distributed for the ensemble "
                "run.",
                "tag": {"ID": "pseudo-claystone-permeability"},
            }
        ],
    }

    with (rock_data_dir / f"{ROCK_NAME}.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(rock_properties, file, sort_keys=False)


def write_site_data():
    """Write `site_data/pseudo_site.yaml`"""
    site_data_dir = DATA_DIR / "site_data"
    site_data_dir.mkdir(parents=True, exist_ok=True)

    site_data = {
        "name": "Pseudo_Site",
        "description": "A made-up single-unit site used to showcase the "
        "YAML/folder-driven yaml2nuctrans workflow.",
        ROCK_NAME: {
            "source": "pseudo",
            "simplified_lithology": ["Mudstone"],
        },
    }

    with (site_data_dir / "pseudo_site.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(site_data, file, sort_keys=False)


def write_geometry_data():
    """Write `geometry/rock_interface.yaml`: a single ~10 m thick unit, top
    at 0 m. The bottom interface is given a uniform distribution.
    """
    geometry_dir = DATA_DIR / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    geometry_data = {
        f"{ROCK_NAME}_top": [
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
                "description": "Top of the pseudo claystone unit, fixed at "
                "the point source location.",
                "tag": {"ID": "pseudo-claystone-top"},
            }
        ],
        f"{ROCK_NAME}_bottom": [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": TOP_ELEVATION - THICKNESS,
                "value_min": TOP_ELEVATION - THICKNESS - 2.0,
                "value_max": TOP_ELEVATION - THICKNESS + 2.0,
                "value_std": 1.15,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": "scipy.stats.uniform(loc=-12.0, scale=4.0).rvs(size=1000000, random_state=21)",
                },
                "unit_str": "m",
                "unit_base": [0, 1, 0, 0, 0, 0, 0],
                "description": "Bottom of the pseudo claystone unit, "
                "uniformly distributed for the ensemble run.",
                "tag": {"ID": "pseudo-claystone-bottom"},
            }
        ],
    }

    with (geometry_dir / "rock_interface.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(geometry_data, file, sort_keys=False)


def write_nuclide_data():
    """Write the four nuclide-data folders for the single nuclide I-129:
    emitted energy, species type, sorption coefficient, and water
    diffusivity. Sorption and diffusivity are both given probability
    distributions.
    """
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
        NUCLIDE_NAME: {
            "source": "ICRP2008",
            "alpha": 0.0,
            "electron": 1.0430169887340001e-14,
            "photon": 4.0374851176800005e-15,
            "total": 1.4451633238680003e-14,
            "unit_str": "kg*m^3/s^2",
            "unit_base": [1, 2, -2, 0, 0, 0, 0],
        }
    }
    with (emitted_energy_dir / "emitted_energy.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(emitted_energy_data, file, sort_keys=False)

    species_type_data = {
        NUCLIDE_NAME: {
            "source": "iaea2009",
            "species_type": "anion",
            "element_category": "halogen",
            "description": "Iodine-129 exists predominantly as the simple "
            "iodide anion.",
        }
    }
    with (species_type_dir / "species_type.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(species_type_data, file, sort_keys=False)

    sorption_data = {
        NUCLIDE_NAME: [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": 0.00181,
                "value_min": 0.0,
                "value_max": 0.0158,
                "value_std": 0.00361,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": "scipy.stats.lognorm(s=1.27, scale=0.000811, loc=0.0).rvs(size=1000000, random_state=21)",
                },
                "unit_str": "m^3/kg",
                "unit_base": [-1, 3, 0, 0, 0, 0, 0],
                "description": "Pseudo sorption coefficient (Kd) for I-129 "
                "in the pseudo claystone unit, lognormally distributed for "
                "the ensemble run.",
                "tag": {"ID": "pseudo-i129-sorption"},
            }
        ]
    }
    with (sorption_dir / f"{ROCK_NAME}.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(sorption_data, file, sort_keys=False)

    diffusivity_data = {
        NUCLIDE_NAME: [
            {
                "type": "scalar",
                "source": "pseudo",
                "value": 1.99e-09,
                "value_min": 1.13e-09,
                "value_max": 3.39e-09,
                "value_std": 2.4e-10,
                "probability_distribution": {
                    "sample_size": 1_000_000,
                    "sampled_data": "scipy.stats.lognorm(s=0.12, scale=1.98e-09, loc=0.0).rvs(size=1000000, random_state=21)",
                },
                "unit_str": "m^2/s",
                "unit_base": [0, 2, -1, 0, 0, 0, 0],
                "description": "Pseudo water diffusivity for I-129, "
                "lognormally distributed for the ensemble run.",
                "tag": {"ID": "pseudo-i129-diffusivity"},
            }
        ]
    }
    with (diffusivity_dir / f"{ROCK_NAME}.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(diffusivity_data, file, sort_keys=False)


def main():
    write_rock_data()
    write_site_data()
    write_geometry_data()
    write_nuclide_data()
    print(f"Pseudo data written to {DATA_DIR}")


if __name__ == "__main__":
    main()

import numpy as np
import h5py
import gzip
import pickle
import yaml
from pathlib import Path

def load_dict_from_hdf5(file_path) -> dict:
    def load_group(group):
        result = {}

        # Load datasets
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[:]

            elif isinstance(item, h5py.Group):
                result[key] = load_group(item)

        # Load attributes
        for key, value in group.attrs.items():
            result[key] = value

        return result

    with h5py.File(file_path, "r") as f:
        return load_group(f)

def load_sampled_data(file_path) -> dict:
    file_path = Path(file_path)
    suffix = file_path.suffixes
    # YAML
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    # Pickle gzip
    elif suffix[-2:] == [".pkl", ".gz"]:
        with gzip.open(file_path, "rb") as f:
            return pickle.load(f)
    # HDF5
    elif file_path.suffix.lower() in {".h5", ".hdf5"}:
        return load_dict_from_hdf5(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. "
            "Expected .yaml, .yml, .pkl.gz, .h5, or .hdf5."
        )

def count_samples(node) -> int:
    """Return ensemble size from the first leaf array found in the tree."""
    if isinstance(node, dict):
        for value in node.values():
            return count_samples(value)
    return int(np.asarray(node).shape[0])

def apply_sample(
    i: int,
    uncertain_parameters: dict,
    geometry_database,
    rock_database,
    nuclide_database,
) -> None:
    """Set run_value of every uncertain parameter to the i-th sampled value."""

    # --- geometry: rock interfaces (e.g. Host_rock_bottom) ---
    for interface_name, values in uncertain_parameters.get("geometry", {}).get(
        "rock_interface", {}
    ).items():
        geometry_database.by_rock_interface[interface_name].run_value = float(
            values[i]
        )

    # --- rock data: density, porosity, ... per rock unit ---
    for rock_name, properties in uncertain_parameters.get("rock_data", {}).items():
        rock = rock_database.by_rock_name.get(rock_name)
        for property_name, values in properties.items():
            rock.get_property(property_name).run_value = float(values[i])

    # --- nuclide water diffusivity: per rock unit, per nuclide ---
    for rock_name, nuclides in uncertain_parameters.get(
        "nuclide_water_diffusivity_data", {}
    ).items():
        for nuclide_name, values in nuclides.items():
            nuclide = nuclide_database.by_nuclide_name.get(nuclide_name)
            nuclide.get_nuclide_water_diffusivity(rock_name).run_value = float(
                values[i]
            )

    # --- sorption coefficients ---
    for rock_name, nuclides in uncertain_parameters.get(
        "nuclide_sorption_data", {}
    ).items():
        
        for nuclide_name, values in nuclides.items():
            nuclide = nuclide_database.by_nuclide_name.get(nuclide_name)
            nuclide.get_sorption_coefficient(rock_name).run_value = float(
                values[i]
            )
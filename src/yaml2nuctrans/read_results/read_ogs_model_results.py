import os
from ogstools.ogs6py import Project
from collections import deque
from pathlib import Path
import copy
import h5py

import numpy as np
import radioactivedecay as rd
import vtuIO
import vtk
from astropy import constants as const

def get_cell_vals(mesh, field_of_interest):
    """Function to read a specific cell data field and store it alongside its corresponding cell geometry (points that define the cell) in a dictionary.

    Args:
        mesh (vtu): vtu file that contains simulation results.
        field_of_interest (str): name of the cell data array to extract.

    Returns:
        dict: a dict keyed by cell index (int), and each value containing 'point_ids', 'point_coords', and 'value'.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh)
    reader.Update()

    output = reader.GetOutput()
    n_cells = output.GetNumberOfCells()
    points = output.GetPoints()

    cell_data = output.GetCellData()
    field = cell_data.GetArray(field_of_interest)
    if field is None:
        raise ValueError(
            f"Field '{field_of_interest}' not found in CellData. "
            "Please make sure the field name exists and that it is stored as CellData."
        )

    n_cells = output.GetNumberOfCells()
    n_component = field.GetNumberOfComponents()  # dimension of the field of interest

    cell_dict = {}
    for cell_i in range(n_cells):

        cell = output.GetCell(cell_i)
        point_ids = [cell.GetPointId(p) for p in range(cell.GetNumberOfPoints())]
        point_coords = [points.GetPoint(id) for id in point_ids]

        field_values = np.zeros(n_component)
        for comp_j in range(n_component):
            field_values[comp_j] = field.GetComponent(cell_i, comp_j)

        cell_dict[cell_i] = {
            "point_ids": point_ids,
            "points": point_coords,
            "value": field_values,
        }

    return cell_dict


def get_points_vals(
    mesh,
    field_of_interest,
    get_field_component_index=[2],
    sort_data=False,
    sort_by_index=2,
):
    """Function to extract points coordinates and output variable values from a vtu file.

    Args:
        mesh (vtu): vtu file that contains simulation results.
        field_of_interest (str): name of the point data array to extract.
        get_field_component_index (list, optional): a list that specifies which components of the field to extract for each process variable. The index is the position of the component to extract. Defaults to [2].
        sort_data (bool, optional): whether to sort the data by point coordinates. Defaults to False.
        sort_by_index (int, optional): index of the coordinate to sort by. Defaults to 2 (z-coordinate).


    Returns:
        coords_vals (dict): dictionary with coordinates as keys and output variable values as values
    """

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh)
    reader.Update()

    output = reader.GetOutput()

    points = output.GetPoints()
    point_data = output.GetPointData()
    field = point_data.GetArray(field_of_interest)

    if field is None:
        raise ValueError(
            f"Field '{field_of_interest}' not found in point data. "
            "Please make sure the field name exists and that it is stored as PointData."
        )

    if field.GetNumberOfComponents() > 1:
        coords_vals = {
            tuple(points.GetPoint(pid)): (
                [field.GetTuple(pid)[i] for i in get_field_component_index]
                if len(get_field_component_index) > 1
                else field.GetTuple(pid)[get_field_component_index[0]]
            )
            for pid in range(output.GetNumberOfPoints())
        }
    else:
        coords_vals = {
            tuple(points.GetPoint(pid)): float(field.GetTuple1(pid))
            for pid in range(output.GetNumberOfPoints())
        }

    if sort_data:
        coords_vals_sorted = sorted(
            coords_vals.items(), key=lambda item: item[0][sort_by_index]
        )
        return dict(coords_vals_sorted)

    return coords_vals


def read_vtu_field(
    vtu_file_path,
    pvd_name,
    field_names,
    get_field_component_index=[2],
    sort_data=False,
    sort_by_index=2,
):
    """Read the concentration of nuclides from a vtu file.

    Args:
        vtu_file_path (str): path to the vtu file.
        pvd_name (str): pvd file name.
        field_names (list): names of the fields.
        get_field_component_index (list, optional): a list that specifies which components of the field to extract for each process variable. The index is the position of the component to extract. Defaults to [2].
        sort_data (bool, optional): whether to sort the data by point coordinates. Defaults to False.
        sort_by_index (int, optional): index of the coordinate to sort by. Defaults to 2 (z-coordinate).

    Returns:
        vals(np.ndarray): values of fields at different time steps and points.
    """
    original_dir = os.getcwd()
    os.chdir(vtu_file_path)
    results_pvd = vtuIO.PVDIO(pvd_name, dim=3)

    timesteps_in_s = results_pvd.timesteps

    vals = {}

    for field in field_names:
        vals[field] = []
        for i in range(len(timesteps_in_s)):
            coords_vals = get_points_vals(
                results_pvd.vtufilenames[i],
                field,
                get_field_component_index=get_field_component_index,
                sort_data=sort_data,
                sort_by_index=sort_by_index,
            )
            for coords, val in coords_vals.items():
                vals[field].append(
                    {
                        "x": float(coords[0]),
                        "y": float(coords[1]),
                        "z": float(coords[2]),
                        "t": float(timesteps_in_s[i]),
                        "val": val,
                    }
                )

    os.chdir(original_dir)
    return vals

def build_result_template(sim_config, nuclide_database) -> dict:
    results = {}
    for QoI in sim_config.quantity_of_interests:
        if QoI == 'concentration':
            for nuclide in nuclide_database.nuclide_names:
                results[nuclide] = {}
        elif QoI == 'Flux':
            for nuclide in nuclide_database.nuclide_names:
                results[nuclide + QoI] = {}
        else:
            results[QoI] = {}
    return results

def merge_sample_results(all_run_results, qoi_results, sample_index) -> dict:
    """Merge one sample's results into the shared container.

    Args:
        all_run_results (dict): Shared container for all samples' results.
        qoi_results (dict): Results of one sample run.
        sample_index (int): Index of the current sample run.
    """
    merged = copy.deepcopy(all_run_results)
    for QoI, entries in qoi_results.items():
        shared = merged[QoI]

        new_keys = entries.keys() - shared.keys()
        missing_keys = shared.keys() - entries.keys()

        # backfilled with None for previous samples
        for key in new_keys:
            shared[key] = [None] * sample_index

        # padded with None
        for key in missing_keys:
            shared[key].append(None)

        for key, values in entries.items():
            shared[key].extend(values)
            
    return merged

def read_sim_results(results_output_dir, sim_config, nuclide_database, get_field_component_index, sort_by_index) -> dict:
    # build the result container ONCE, before the loop
    qoi_results = build_result_template(sim_config, nuclide_database)

    field_results = read_vtu_field(
        vtu_file_path=results_output_dir,
        pvd_name=f"{sim_config.project_name}.pvd",
        field_names=qoi_results.keys(),
        get_field_component_index=get_field_component_index,
        sort_data=True,   # sort by depth in z direction
        sort_by_index=sort_by_index,
    )

    for QoI in qoi_results.keys():
        for space_time_val in field_results[QoI]:
            key = tuple(space_time_val[c] for i, c in enumerate(("x", "y", "z")) if i in get_field_component_index) + (space_time_val["t"],)
            qoi_results[QoI].setdefault(key, []).append(space_time_val["val"])

    return qoi_results

def save_dict_to_group(group: h5py.Group, d: dict) -> None:
    """Recursively write a nested {key: array} dict as HDF5 groups/datasets."""
    for key, value in d.items():
        if isinstance(value, dict):
            save_dict_to_group(group.create_group(key), value)
        else:
            group.create_dataset(key, data=np.asarray(value))


def save_results_to_group(results: dict, parent: h5py.Group) -> None:
    """Writing into an existing group."""
    for qoi_name, qoi_dict in results.items():
        grp = parent.create_group(qoi_name)

        keys = list(qoi_dict.keys())
        n_keys = len(keys)

        key_arr = np.array(keys, dtype=np.float64)
        grp.create_dataset("keys", data=key_arr)

        n_samples = next(len(vals) for vals in qoi_dict.values() if vals is not None)

        sample_val = next(
            v for vals in qoi_dict.values() if vals is not None
            for v in vals if v is not None
        )
        is_vector = isinstance(sample_val, (list, tuple, np.ndarray))

        if not is_vector:
            data = np.full((n_keys, n_samples), np.nan, dtype=np.float64)
            for i, k in enumerate(keys):
                for j, v in enumerate(qoi_dict[k]):
                    if v is not None:
                        data[i, j] = float(v)
        else:
            # vector QoI: (n_keys, n_samples, n_comp)
            n_comp = len(sample_val)
            data = np.full((n_keys, n_samples, n_comp), np.nan, dtype=np.float64)
            for i, k in enumerate(keys):
                for j, v in enumerate(qoi_dict[k]):
                    if v is not None:
                        data[i, j, :] = np.asarray(v, dtype=np.float64)

        grp.create_dataset("values", data=data)
        grp.attrs["n_keys"] = n_keys
        grp.attrs["n_samples"] = n_samples
        grp.attrs["n_key_dims"] = key_arr.shape[1]
        grp.attrs["is_vector"] = bool(is_vector)


def save_results_to_hdf5(all_run_results: dict, h5_path: str, sampled_data: dict | None = None, save_sampled_data: bool = False) -> None:
    """Write sampled input parameters and simulation results into ONE HDF5 file."""
    with h5py.File(h5_path, "w") as f:
        if save_sampled_data and sampled_data is not None:
            # inputs
            save_dict_to_group(f.create_group("sampled_data"), sampled_data)
        # outputs
        save_results_to_group(all_run_results, f.create_group("simulation_results"))


def load_results_from_hdf5(h5_path: str) -> dict:
    """
    Load results saved by save_results_to_hdf5.
    """
    def read_group(group):
        """Recursively read an HDF5 group into a nested dict of arrays."""
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                data[key] = item[...]
            else:
                data[key] = read_group(item)
        return data

    loaded = {}

    with h5py.File(h5_path, "r") as f:
        # --- sampled inputs ---
        if "sampled_data" in f:
            loaded["sampled_data"] = read_group(f["sampled_data"])

        # --- simulation results ---
        results = {}
        for qoi_name in f["simulation_results"].keys():
            grp = f["simulation_results"][qoi_name]

            # native Python floats for keys
            keys = [tuple(float(v) for v in row) for row in grp["keys"][:]]
            data = grp["values"][:]
            is_vector = bool(grp.attrs["is_vector"])

            qoi_dict = {}
            for i, k in enumerate(keys):
                if not is_vector:
                    qoi_dict[k] = [None if np.isnan(v) else float(v) for v in data[i, :]]
                else:
                    qoi_dict[k] = [
                        None if np.all(np.isnan(data[i, j])) else data[i, j].tolist()
                        for j in range(data.shape[1])
                    ]
            results[qoi_name] = qoi_dict

        loaded["simulation_results"] = results

    return loaded
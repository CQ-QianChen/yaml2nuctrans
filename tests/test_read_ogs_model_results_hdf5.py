import numpy as np
import pytest

from yaml2nuctrans.read_results.read_ogs_model_results import (
    load_results_from_hdf5,
    save_results_to_hdf5,
)

DOMAIN_ZONE = "domain"
HOST_ROCK_ZONE = "host_rock"


@pytest.mark.parametrize(
    "all_run_results, key_types",
    [
        pytest.param(
            {
                # concentration: keyed by (z, t), both numeric
                "I-129": {
                    (0.0, 100.0): [1.0, 2.0],
                    (1.0, 200.0): [1.5, None],  # missing sample backfilled with None
                },
            },
            (float, float),
            id="numeric_keys",
        ),
        pytest.param(
            {
                # ewg-discharge QoIs: keyed by (zone_name, t), zone_name a string
                "total_discharged": {
                    (DOMAIN_ZONE, 0.0): [10.0],
                    (HOST_ROCK_ZONE, 0.0): [0.0],
                    (DOMAIN_ZONE, 50.0): [8.0],
                    (HOST_ROCK_ZONE, 50.0): [2.0],
                },
            },
            (str, float),
            id="string_zone_label_keys",
        ),
    ],
)
def test_scalar_results_round_trip(tmp_path, all_run_results, key_types):
    """Scalar (non-vector) QoI values should round-trip exactly."""
    h5_path = tmp_path / "results.h5"
    save_results_to_hdf5(all_run_results, str(h5_path))
    loaded = load_results_from_hdf5(str(h5_path))

    for qoi_name, qoi_dict in all_run_results.items():
        loaded_qoi = loaded["simulation_results"][qoi_name]
        assert loaded_qoi == qoi_dict
        for key in loaded_qoi:
            assert tuple(type(component) for component in key) == key_types


def test_vector_results_round_trip(tmp_path):
    """Test vector-valued QoI entries (e.g. multi-component fields)."""
    all_run_results = {
        "flux": {
            (0.0, 0.0, 0.0, 0.0): [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            (0.0, 0.0, 1.0, 0.0): [[7.0, 8.0, 9.0], None],
        },
    }

    h5_path = tmp_path / "results.h5"
    save_results_to_hdf5(all_run_results, str(h5_path))
    loaded = load_results_from_hdf5(str(h5_path))

    loaded_qoi = loaded["simulation_results"]["flux"]
    for key, values in all_run_results["flux"].items():
        assert loaded_qoi[key] == values


def test_all_requested_qois_round_trip_together(tmp_path):
    """Test mixed OoI values loads back correctly."""
    nuclide = "I-129"
    all_run_results = {
        # concentration: scalar, keyed by (z, t)
        nuclide: {
            (0.0, 0.0): [1.0, 1.2],
            (1.0, 0.0): [0.8, None],
        },
        # Flux: vector-valued, keyed by (z, t)
        f"{nuclide}Flux": {
            (0.0, 0.0): [[1.0, 0.0, -1.0], [1.1, 0.0, -1.1]],
            (1.0, 0.0): [[0.5, 0.0, -0.5], None],
        },
        # per-nuclide ewg-discharge fields: scalar, keyed by (zone_name, t)
        f"{nuclide}_discharged": {
            (DOMAIN_ZONE, 0.0): [0.0, 0.0],
            (DOMAIN_ZONE, 50.0): [2.0, 2.5],
        },
        f"{nuclide}_frac_discharged": {
            (DOMAIN_ZONE, 0.0): [0.0, 0.0],
            (DOMAIN_ZONE, 50.0): [0.2, 0.25],
        },
        f"{nuclide}_release_rate": {
            (DOMAIN_ZONE, 0.0): [0.0, 0.0],
            (DOMAIN_ZONE, 50.0): [0.04, 0.05],
        },
        # total_* aggregates: scalar, keyed by (zone_name, t)
        "total_discharged": {
            (DOMAIN_ZONE, 0.0): [0.0, 0.0],
            (DOMAIN_ZONE, 50.0): [2.0, 2.5],
        },
        "total_frac_discharged": {
            (DOMAIN_ZONE, 0.0): [0.0, 0.0],
            (DOMAIN_ZONE, 50.0): [0.2, 0.25],
        },
        "total_release_rate": {
            (DOMAIN_ZONE, 0.0): [0.0, 0.0],
            (DOMAIN_ZONE, 50.0): [0.04, 0.05],
        },
    }

    h5_path = tmp_path / "results.h5"
    save_results_to_hdf5(all_run_results, str(h5_path))
    loaded = load_results_from_hdf5(str(h5_path))

    assert set(loaded["simulation_results"].keys()) == set(all_run_results.keys())
    for qoi_name, qoi_dict in all_run_results.items():
        assert loaded["simulation_results"][qoi_name] == qoi_dict

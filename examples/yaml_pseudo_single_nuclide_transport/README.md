# Example : YAML/folder-driven single-nuclide transport simulation

A one-dimensional, single-nuclide (I-129) transport example driven
entirely from YAML configuration and a folder of YAML data files  —
instead of hand-built Python objects (compare with
[`../decay_chain_ogs_vs_analytical/`](../decay_chain_ogs_vs_analytical/)).

## Scenario

- A single ~10 m thick pseudo claystone unit, point source at 0 m (the
  unit's top).
- Dirichlet concentration = 0.0 at the source.
- Initial concentration = 1.0 everywhere else.
- Decay, diffusion, advection, and sorption all enabled.
- The unit's bottom interface, porosity,
  density, intrinsic permeability, the I-129 sorption coefficient, and
  the I-129 water diffusivity are all given probability distributions
  for the ensemble run.

## Files

- `create_pseudo_data.py` — writes the pseudo rock, site, geometry, and
  nuclide data into `data/` (the folder layout read by
  `RockData.from_folder`, `NuclideData.from_yaml_files`, and
  `GeometryData.from_folder`). 
- `config_point_source.yaml` — the `SimulationConfig` YAML: source
  location, initial/boundary conditions, pressure setup, enabled
  processes, and the OGS time-stepping/solver setup.
- `sample_config.yaml` — the `uncertain_parameters` spec consumed by the
  sampling CLI (`python -m yaml2nuctrans.sampling.sampling_func`): the
  pseudo unit's bottom interface, porosity, density, intrinsic
  permeability, and I-129's sorption coefficient and water diffusivity.
- `make_ensemble_samples.sh` — draws a small (size 8) Sobol' ensemble over
  the six uncertain parameters via the sampling CLI entry point and saves
  it as `data/sampled_data/sample_<method>_size<size>_seed<seed>.h5`.
- `run_single.sh` — runs the deterministic, single best-estimate
  simulation via the CLI entry point
  (`python -m yaml2nuctrans.get_model.ogs_model`). Writes the
  results HDF5 file to: `output/single_run/single_best_estimate_results.h5`.
- `run_ensemble.sh` — runs the 8-member ensemble via the same CLI entry
  point (calling `make_ensemble_samples.sh` first if none is found). Writes the results HDF5 file to: `output/ensemble_run/ensemble_results.h5`.

- `single_nuclide_transport.ipynb` — a notebook front end that writes the
  pseudo data, runs `run_single.sh` and `run_ensemble.sh` as shell
  commands, and plots concentration profiles.

## Usage

```console
python create_pseudo_data.py
./run_single.sh
./make_ensemble_samples.sh   # once, to (re)write data/sampled_data/
./run_ensemble.sh            # generates the ensemble samples itself if missing
```

Each run script writes its results HDF5 file to `output/single_run/` or
`output/ensemble_run/`.

Or, to run the same steps interactively with inline plots, open
`single_nuclide_transport.ipynb`.

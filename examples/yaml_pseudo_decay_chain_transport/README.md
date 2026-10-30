# Example : YAML/folder-driven decay-chain transport simulation

A one-dimensional, decay-chain (Cm-247, Am-243, Pu-239, U-235,
Pa-231, Ac-227) transport example driven entirely from YAML configuration
and a folder of YAML data files (compare with
[`../decay_chain_ogs_vs_analytical/`](../decay_chain_ogs_vs_analytical/) using hand-built Python objects).

## Scenario

- Three stacked pseudo rock units: Pseudo_Sandstone (0 to -10 m),
  Pseudo_Claystone (-10 to -25 m), and Pseudo_Limestone (-25 to -45 m).
- A 2.5 m line-source repository centered in the Pseudo_Claystone host
  rock (-16.25 to -13.75 m), holding a per-nuclide inventory (32,770 mol
  Cm-247 down to 1,000 mol Ac-227) over a 3000 m x 3000 m
  repository area. The initial pore-water concentration in the source
  zone for each nuclide is computed automatically from its inventory via
  `initial_condition_nuclide.source_concentration`.
- Dirichlet concentration = 0.0 at the top boundary, for every nuclide.
- Decay chain: Cm-247 - Am-243 - Pu-239 - U-235 -
  Pa-231 -> Ac-227, diffusion, advection, and sorption all enabled.
- `repository_discharge.host_rock: Pseudo_Claystone` treats
  Pseudo_Claystone as the containment-providing zone (ewg), with
  Pseudo_Sandstone and Pseudo_Limestone as the zones material is
  discharged into.
- Pseudo_Claystone unit's bottom interface, porosity, density, intrinsic
  permeability, and every chain member's sorption coefficient and water
  diffusivity are all given probability distributions for the ensemble
  run.

## Files

- `create_pseudo_data.py` — writes the pseudo rock, site, geometry, and
  nuclide data into `data/` (the folder layout read by
  `RockData.from_folder`, `NuclideData.from_yaml_files`, and
  `GeometryData.from_folder`). 
- `config_line_source.yaml` — the `SimulationConfig` YAML: source
  location, initial/boundary conditions, pressure setup, enabled
  processes, and the OGS time-stepping/solver setup.
- `sample_config.yaml` — the `uncertain_parameters` spec consumed by the
  sampling CLI (`python -m yaml2nuctrans.sampling.sampling_func`): the
  pseudo unit's bottom interface, porosity, density, intrinsic
  permeability, and every chain member's sorption coefficient and water
  diffusivity.
- `make_ensemble_samples.sh` — draws a small (size 8) Sobol' ensemble over
  the 16 uncertain parameters via the sampling CLI entry point and saves
  it as `data/sampled_data/sample_<method>_size<size>_seed<seed>.h5`.
- `run_single.sh` — runs the deterministic, single best-estimate
  simulation via the CLI entry point
  (`python -m yaml2nuctrans.get_model.ogs_model`). Writes the
  results HDF5 file to: `output/single_run/single_best_estimate_results.h5`.
- `run_ensemble.sh` — runs the 8-member ensemble via the same CLI entry
  point (calling `make_ensemble_samples.sh` first if none is found). Writes the results HDF5 file to: `output/ensemble_run/ensemble_results.h5`.
- `decay_chain_transport.ipynb` — a notebook front end that writes the
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
`decay_chain_transport.ipynb`.

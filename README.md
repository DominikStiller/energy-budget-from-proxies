# Top-of-atmosphere radiation fields over the last millennium reconstructed from proxies

This repository hosts the code for Stiller & Hakim (2025), "Top-of-atmosphere radiation fields over the last millennium reconstructed from proxies". The preprint is [available online](https://arxiv.org/abs/2510.09896). For a detailed description of the algorithms used for the reconstruction (LIM, EnKF, PSMs), refer to the supplement.

> Earth's energy imbalance at the top of the atmosphere is a key climate system metric, but its variability is poorly constrained by the short observational record and large uncertainty in coupled climate models. While existing ocean heat content reconstructions offer a longer perspective, they cannot separate the contributions of shortwave and longwave radiation, obscuring the underlying processes. We extend the energy budget record by reconstructing the top-of-atmosphere radiation and related surface variables over the last millennium (850--2000 CE). Our method combines proxy data and model dynamics using seasonal, gridded data assimilation, exploiting the covariance of radiation with proxies sensitive to surface temperatures. The method validates well in a pseudoproxy experiment and against instrumental observations. We find that a last-millennium cooling trend coincides with heat loss that gradually slowed, although there are intermittent multidecadal periods of energy gain. The cooling trend is associated with a southwestward shift of Indo--Pacific convection and growth of sea ice, with seasonal sea ice trends following orbital-scale changes in polar insolation. We also find that the upper-ocean heat content following large volcanic eruptions does not begin to recover until 5--10 years later, suggesting the initiation of the Little Ice Age by decadally-paced eruptions in the early 1100s and late 1200s. Indeed, the latter period marks the decade of largest energy loss over the last millennium. Our reconstruction reveals that the energy imbalance for all 20-yr periods after 1980 is unprecedented in the pre-industrial period.

## Code structure
 * `lmrecon/`: Reconstruction code
   * `lmrecon/scripts/`: Executable scripts (see "Running the reconstruction" below)
   * `lmrecon/reconstruction.py`: Entry point for all reconstructions
   * `lmrecon/da.py`: Implements online data assimilation
   * `lmrecon/kf.py`: Implements ensemble Kalman filter (EnKF)
   * `lmrecon/mapper.py`: Implements physical space–EOF space mapper
   * `lmrecon/psm.py`: Implements proxy system models (PSM)
   * `lmrecon/lim.py`: Implements linear inverse model (LIM)
 * `notebooks/`: Analysis and prototyping notebooks
   * `notebooks/figures_paper_StillerHakim2025.ipynb`: Figures for Stiller & Hakim (2025)
 * `jobs/`: PBS jobs for NCAR HPC
 * `pyproject.toml`: Python environment configuration, uses Pixi as package manager


## Running the reconstruction
*This list does not include comprehensive instructions to reproduce the results, rather it serves as an outline of the process. All scripts are located in `lmrecon/scripts/`. For some steps, PBS job scripts are available in `jobs/`.*
1. Install the [Pixi package manager](https://pixi.sh/dev/), then run `pixi install`.
2. Modify `get_*_path()` in `lmrecon/util.py` to point to your local directories.
3. Download the CMIP6 past1000 simulations from ESGF (e.g., using [esgpull](https://github.com/ESGF/esgf-download/tree/main) or `download_cmip6_dataset.py`).
4. Download the proxy data (e.g., Pages2k or CoralHydro2k).
5. Download the instrumental PSM calibration data (e.g., GISTEMP and ERSST).
6. Update the `intake-esm` catalog to include the downloaded CMIP6 simulations using `update_intake_catalog.py`.
7. Compute seasonal, detrended anomalies using `compute_seasonal_averages.py`, then `compute_seasonal_anomalies.py`, then `compute_seasonal_detrended_anomalies.py`. This also regrids the simulations to the common 2°×2° grid.
8. Fit the mapper from the physical space to the EOF space using `fit_mapper.py`. This also produces the LIM training data.
9. Regrid and seasonally average the instrumental data using the commented-out code in `datasets.py`.
10. Combine the proxy databases using `assemble_pdb.py`. This also removes duplicates.
11. Calibrate the PSMs using `calibrate_psms.py`. This excludes proxies that have insufficient calibration correlations.
12. Compute the sea ice concentration climatology using `compute_hybrid_siconc_climatology.py`. This requires the climatology from Cooper et al. (2025).
13. Run the reconstruction using `run_reconstruction_allproxies.py` (assimilates all proxies) or using `run_reconstruction_single.py` (assimilates a subset of proxies for Monte Carlo iterations). This produces the reconstruction in EOF space.
14. Postprocess the reconstruction using `postprocess_reconstruction.py`. This maps the reconstruction from EOF space into the physical space and computes averages. If used in the previous step, multiple Monte Carlo iterations can be combined using `postprocess_reconstruction_mc.py`.
15. Analyze the reconstruction using `notebooks/figures_paper_StillerHakim2025.ipynb` or `notebooks/compare_reconstructions.ipynb`.

For each model prior (CMIP6 simulation used to train the LIM), the following steps need to be repeated: fit mapper, calibrate PSM, run reconstruction.

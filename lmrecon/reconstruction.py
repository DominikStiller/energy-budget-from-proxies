from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import cfr
import numpy as np
import xarray as xr
from cfr import ProxyDatabase

from lmrecon.da import (
    LIMandTimeWindowEnSRFinReducedSpaceDataAssimilation,
    create_initial_ensemble_from_sample,
)
from lmrecon.logger import get_logger, logging_disabled
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.scripts.train_lim import train_lim
from lmrecon.time import Season
from lmrecon.util import get_data_path, stack_state, to_math_order

if TYPE_CHECKING:
    from lmrecon.psm import PSM

logger = get_logger(__name__)


def sample_assimilated_proxies(
    pdb: cfr.ProxyDatabase, assimilated_fraction: float, seed_offset: int
) -> tuple[cfr.ProxyDatabase, cfr.ProxyDatabase]:
    records = sorted(pdb.records.items())
    np.random.default_rng(seed=544502 + seed_offset).shuffle(records)

    idx_last_assimilated = int(len(records) * assimilated_fraction)
    return cfr.ProxyDatabase(dict(records[:idx_last_assimilated])), cfr.ProxyDatabase(
        dict(records[idx_last_assimilated:])
    )


def run_reconstruction(
    output_dir: Path,
    year_start: int,
    year_end: int,
    n_ens: int,
    obs_dataset: str,
    assimilated_fraction: float | None = None,
    seed_offset: int = 0,
    window_mode: str = "average",
    removed_proxies: list[str] | None = None,
):
    (output_dir / "prior").mkdir(parents=True)
    (output_dir / "posterior").mkdir(parents=True)

    logger.info(f"Saving to {output_dir}")

    obs_path = get_data_path() / obs_dataset
    with (obs_path / "metadata.json").open() as f:
        obs_metadata = json.load(f)
        mapper_id = obs_metadata["mapper_id"]

    logger.info(f"Loading proxy database and PSMs from {obs_path}")
    pdb: ProxyDatabase = pickle.load((obs_path / "pdb.pkl").open("rb"))

    if removed_proxies:
        for pid in removed_proxies:
            del pdb.records[pid]

    if assimilated_fraction:
        pdb_assimilated, pdb_withheld = sample_assimilated_proxies(
            pdb, assimilated_fraction, seed_offset
        )
    else:
        pdb_assimilated = pdb
        pdb_withheld = None

    # Pre-slice, then remove proxies that do not overlap with reconstruction period
    pdb_assimilated = pdb_assimilated.slice((year_start - 1, year_end + 2))
    pdb_assimilated = cfr.ProxyDatabase(
        {pid: pobj for pid, pobj in pdb_assimilated.records.items() if len(pobj.time) > 0}
    )

    psms: dict[str, PSM | dict[Season, PSM]] = pickle.load((obs_path / "psms.pkl").open("rb"))
    psms = {pid: psm for pid, psm in psms.items() if pid in pdb_assimilated.records}
    assert set(pdb_assimilated.records.keys()) == set(psms.keys())

    lim = train_lim(mapper_id)
    tau = lim.tau
    assert np.isclose(tau, 0.25)  # only support seasonal

    mapper_path = get_data_path() / "mapper" / mapper_id
    mapper = PhysicalSpaceForecastSpaceMapper.load(mapper_path / "mapper.pkl")
    with (mapper_path / "metadata.json").open() as f:
        mapper_metadata = json.load(f)
        # Physical dataset may be detrended
        initial_prior_dataset = Path(mapper_metadata["physical_dataset"]).with_name(
            "seasonal_anomalies.zarr"
        )

    json.dump(
        {
            "obs_dataset": str(obs_dataset),
            "mapper_id": mapper_id,
            "initial_prior_dataset": str(initial_prior_dataset),
            "year_start": year_start,
            "year_end": year_end,
            "n_ens": n_ens,
            "assimilated_fraction": assimilated_fraction,
            "seed_offset": seed_offset,
            "window_mode": window_mode,
            "removed_proxies": removed_proxies,
        },
        (output_dir / "metadata.json").open("w"),
        indent=4,
    )

    pickle.dump(pdb_assimilated, (output_dir / "pdb_assimilated.pkl").open("wb"))
    if pdb_withheld:
        pickle.dump(pdb_withheld, (output_dir / "pdb_withheld.pkl").open("wb"))

    logger.info(f"Loading initial ensemble from {initial_prior_dataset}")
    prior = to_math_order(
        stack_state(
            create_initial_ensemble_from_sample(
                xr.open_zarr(get_data_path() / initial_prior_dataset),
                n_ens,
                year_start,
                Season.DJF,
                seed_offset,
            )
        )
    ).compute()

    da = LIMandTimeWindowEnSRFinReducedSpaceDataAssimilation(
        output_dir, pdb_assimilated, psms, mapper, lim, window_mode
    )
    with logging_disabled():
        prior = mapper.forward(prior)

    logger.info("Starting DA cycle")
    da.cycle(prior, year_start, year_end, tau)

    logger.info(f"Saved to {output_dir}")

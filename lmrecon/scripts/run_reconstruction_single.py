from __future__ import annotations

import sys

from lmrecon.logger import get_logger
from lmrecon.reconstruction import run_reconstruction
from lmrecon.scripts.compute_verification_proxies import (
    compute_verification_proxies,
)
from lmrecon.util import get_data_path

logger = get_logger(__name__)


if __name__ == "__main__":
    year_start = 840
    year_end = 2001
    n_ens = 400
    obs_dataset = sys.argv[1]
    if "/" not in obs_dataset:
        obs_dataset = f"obs/{obs_dataset}"
    assimilated_fraction = 0.8
    seed_offset = int(sys.argv[3])
    window_mode = "average"
    removed_proxies = []

    run_id = sys.argv[2]
    output_dir = get_data_path() / "reconstructions" / run_id / str(seed_offset)

    logger.info(f"Starting reconstruction with offset {seed_offset}")
    run_reconstruction(
        output_dir,
        year_start,
        year_end,
        n_ens,
        obs_dataset,
        assimilated_fraction,
        seed_offset,
        window_mode,
        removed_proxies,
    )

    compute_verification_proxies(output_dir)

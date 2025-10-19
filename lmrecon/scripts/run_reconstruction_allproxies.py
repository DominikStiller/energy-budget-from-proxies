from __future__ import annotations

import sys

from lmrecon.logger import get_logger
from lmrecon.reconstruction import run_reconstruction
from lmrecon.util import get_data_path, get_timestamp

logger = get_logger(__name__)


if __name__ == "__main__":
    year_start = 840
    year_end = 2001
    n_ens = 400
    obs_dataset = sys.argv[1]
    if "/" not in obs_dataset:
        obs_dataset = f"obs/{obs_dataset}"
    window_mode = "average"
    removed_proxies = []

    run_id = get_timestamp()
    output_dir = get_data_path() / "reconstructions" / run_id

    run_reconstruction(
        output_dir,
        year_start,
        year_end,
        n_ens,
        obs_dataset,
        window_mode=window_mode,
        removed_proxies=removed_proxies,
    )

    # print()
    # logger.info("Postprocessing reconstruction")
    # postprocess_reconstruction(run_id)

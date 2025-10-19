from __future__ import annotations

import sys

from lmrecon.io import open_mfdataset
from lmrecon.lim import LIM
from lmrecon.logger import get_logger
from lmrecon.util import (
    get_data_path,
    to_math_order,
)

logger = get_logger(__name__)


def train_lim(mapper_id):
    training_data_path = get_data_path() / "mapper" / mapper_id
    data = open_mfdataset(training_data_path / "seasonal_anomalies")["data"]
    # In case we use detrended data
    data = data - data.mean("time")

    logger.info(
        f"Training LIM on {mapper_id} ({len(data.time)} samples, state size {len(data.state)})"
    )

    assert (data.mean("time") < 1e-5).all()

    lim = LIM()
    lim.fit(to_math_order(data))
    return lim


if __name__ == "__main__":
    mapper_id = sys.argv[1]
    train_lim(mapper_id).save(
        get_data_path() / "lim",
        {
            "mapper_id": str(mapper_id),
        },
    )

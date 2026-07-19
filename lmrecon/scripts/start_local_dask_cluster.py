from __future__ import annotations

from threading import Event

import dask
from dask.distributed import LocalCluster

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=dask.system.CPU_COUNT, threads_per_worker=1)
    dashboard = getattr(cluster, "dashboard_link", "unknown")
    scheduler = getattr(cluster, "scheduler_address", None)
    if scheduler is None:
        sched_obj = getattr(cluster, "scheduler", None)
        scheduler = getattr(sched_obj, "address", "unknown") if sched_obj is not None else "unknown"
    print(
        f"Local Dask cluster running at {dashboard} (scheduler: {scheduler}) — press Ctrl+C to exit"
    )
    try:
        Event().wait()  # block until interrupted
    except KeyboardInterrupt:
        pass
    finally:
        cluster.close()

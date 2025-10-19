from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from pyesgf.search import SearchConnection
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from lmrecon.scripts.update_intake_catalog import update_intake_catalog
from lmrecon.util import get_base_path

os.environ["ESGF_PYCLIENT_NO_FACETS_STAR_WARNING"] = "1"


def collect_download_urls(model, experiment, variables, variant="r1i1p1f1", data_node="llnl"):
    if not isinstance(variables, list):
        variables = [variables]

    data_node = {
        "llnl": "aims2.llnl.gov",
        "llnl-esgf": "esgf-data1.llnl.gov",
        "diasjp2": "esgf-data02.diasjp.net",
        "diasjp3": "esgf-data03.diasjp.net",
        "dkrz": "esgf3.dkrz.de",
        "ceda": "esgf.ceda.ac.uk",
        "liu": "esg-dn2.nsc.liu.se",
        "nci": "esgf.nci.org.au",
    }[data_node]

    conn = SearchConnection("https://esgf.ceda.ac.uk/esg-search", distrib=True)

    urls = {}
    for var in variables:
        ctx = conn.new_context(
            project="CMIP6",
            source_id=model,
            experiment_id=experiment,
            variable=var,
            frequency="mon",
            variant_label=variant,
            grid_label="gn",  # always get native grid since we regrid anyways
            data_node=data_node,
            latest="true",
            facets="project,experiment_id",
        )

        if ctx.hit_count == 0:
            raise ValueError(f"No results for variable {var} on {data_node}")

        results = ctx.search()
        versions = {result.json["version"]: result for result in results}
        latest_result = versions[max(versions.keys())]

        print(f"Collecting download URLs for {latest_result.dataset_id}")
        directory = latest_result.json["instance_id"].replace(".", "/")
        urls[directory] = [file.download_url for file in latest_result.file_context().search()]

    return urls


def _download_single_file(i, url, folder, delay=0):
    file = folder / Path(url).name
    if file.exists():
        print(f"[{i + 1}/{n_files}] {file} exists")
        return

    try:
        s = Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[413, 429, 502, 503])
        s.mount("http://", HTTPAdapter(max_retries=retries))
        s.mount("https://", HTTPAdapter(max_retries=retries))

        response = s.get(url, stream=True)
        response.raise_for_status()

        with file.open("wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"[{i + 1}/{n_files}] Downloaded {file}")
    except Exception as e:
        print(f"Failed to download {file.name}. Error: {e!s}")
        file.unlink(missing_ok=True)

    time.sleep(delay)


def download_files_in_parallel(url_dict, max_workers=3, delay=0.5):
    # Use ThreadPoolExecutor to download files in parallel
    with ThreadPoolExecutor(max_workers) as executor:
        futures = []
        i = 0
        for folder, urls in url_dict.items():
            folder = get_base_path() / folder
            folder.mkdir(parents=True, exist_ok=True)

            for url in urls:
                futures.append(executor.submit(_download_single_file, i, url, folder, delay))
                i += 1

        wait(futures)


def download_files_in_sequence(url_dict, delay=0.1):
    # Alternative to download in sequence since there seems to be a rate limit
    i = 0
    for folder, urls in url_dict.items():
        folder = get_base_path() / folder
        folder.mkdir(parents=True, exist_ok=True)

        for url in urls:
            _download_single_file(i, url, folder, delay)
            i += 1


if __name__ == "__main__":
    # model = "CESM2"
    # model = "MRI-ESM2-0"
    # model = "MPI-ESM1-2-LR"
    model = "MIROC-ES2L"
    # experiment = "piControl"
    # experiment = "historical"
    experiment = "past1000"
    variables = ["tas", "tos", "rsdt", "rsut", "rlut", "thetao", "siconc"]
    # variables = ["rsut"]
    variant = "r1i1000p1f2"

    urls_to_download = collect_download_urls(
        model, experiment, variables, variant, data_node="diasjp2"
    )
    n_files = sum(map(len, urls_to_download.values()))

    print(f"Starting downloads ({n_files} files)")
    download_files_in_parallel(urls_to_download)
    # download_files_in_sequence(urls_to_download)

    update_intake_catalog()

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import cfr
import numpy as np
import pandas as pd
from cfr import ProxyDatabase
from pylipd.lipd import LiPD
from tqdm import tqdm

from lmrecon.logger import get_logger
from lmrecon.util import get_base_path, get_spherical_distance

logger = get_logger(__name__)


def load_pages2k():
    lipd = LiPD()
    lipd.load_from_dir(str(get_base_path() / "datasets" / "proxies" / "Pages2k"), parallel=True)

    df = lipd.get_timeseries(lipd.get_all_dataset_names(), to_dataframe=True)[1]

    df = df[df["paleoData_pages2kID"].notnull()]
    df = df[~df["paleoData_pages2kID"].str.startswith("X")]
    df["pid"] = "pages2k_" + df["paleoData_pages2kID"]

    return ProxyDatabase().from_df(df, pid_column="pid")


def load_ch2k():
    # CoralHydro2k
    lipd = LiPD()
    lipd.load_from_dir(
        str(get_base_path() / "datasets" / "proxies" / "CoralHydro2k"), parallel=True
    )

    df = lipd.get_timeseries(lipd.get_all_dataset_names(), to_dataframe=True)[1]

    # Remove age and uncertainty timeseries
    df = df[df["paleoData_variableName"].isin(["d18O", "Sr/Ca"])]
    # Remove d18O of seawater and annually averaged versions
    df = df[
        ~df["paleoData_TSid"].str.contains("_sw") & ~df["paleoData_TSid"].str.contains("_annual")
    ]
    assert df["paleoData_TSid"].is_unique
    df["pid"] = "ch2k_" + df["paleoData_TSid"]

    pdb = ProxyDatabase().from_df(
        df,
        archive_type_column="archiveType",
        proxy_type_column="paleoData_variableName",
        ptype_column="DOES_NOT_EXIST",
        pid_column="pid",
    )
    # Ensure we added all proxies
    assert len(pdb.records) == len(df)
    return pdb


def load_noaa_proxy(file):
    with open(file) as f:
        lines = f.readlines()
    metadata = {}
    for line in lines:
        if line.startswith("#"):
            key_value = line[1:].split(":", 1)  # Split on the first colon
            if len(key_value) == 2:
                key, value = key_value
                metadata[key.strip()] = value.strip()
        else:
            break  # Stop after header ends

    column_metadata = {}
    for line in lines:
        if line.startswith("##"):
            key_value = line[2:].split("\t", 1)  # Split on the first colon
            if len(key_value) == 2:
                key, value = key_value
                column_metadata[key.strip()] = value.strip().split(",")
        elif not line.startswith("#"):
            break  # Stop after header ends

    variable = list(column_metadata.keys())[1]
    data = pd.read_csv(file, delim_whitespace=True, comment="#")
    years = data["age"].values
    values = data[variable].values

    assert metadata["Time_Unit"] == "Year CE"
    archive = {
        "Corals and Sclerosponges": "coral",
    }[metadata["Archive"]]
    value_unit = {
        "per mil VPDB": "permil",
    }[column_metadata[variable][3].strip()]
    pid = f"noaa_{metadata['Collection_Name']}"
    ptype = cfr.proxy.get_ptype(archive, variable)
    lat = (float(metadata["Northernmost_Latitude"]) + float(metadata["Southernmost_Latitude"])) / 2
    lon = (float(metadata["Westernmost_Longitude"]) + float(metadata["Easternmost_Longitude"])) / 2
    elev = float(metadata["Elevation"])

    return cfr.ProxyRecord(
        pid=pid,
        time=years,
        value=values,
        lat=lat,
        lon=lon,
        elev=elev,
        ptype=ptype,
        value_name=variable,
        value_unit=value_unit,
        time_name="Time",
        time_unit="yr",
    )


def _is_duplicate(k1, v1, k2, v2, test_identity, correlation_threshold):
    if v1.ptype != v2.ptype:
        return
    if not np.isclose(v1.dt, v2.dt):
        return
    if get_spherical_distance(v1.lat, v1.lon, v2.lat, v2.lon) > 1000:
        return

    if test_identity:
        if len(v1.value) != len(v2.value):
            return
        if np.isclose(v1.value, v2.value).all():
            return (k1, k2)
    else:
        ts1 = pd.DataFrame({"value": v1.value, "time": v1.time}).set_index("time")
        ts2 = pd.DataFrame({"value": v2.value, "time": v2.time}).set_index("time")

        # Align the two time series
        aligned = pd.merge(
            ts1,
            ts2,
            left_index=True,
            right_index=True,
            how="inner",
            suffixes=("_ts1", "_ts2"),
        )

        if len(aligned) >= 10:
            # Compute correlation
            correlation = aligned["value_ts1"].corr(aligned["value_ts2"])
            if correlation > correlation_threshold:
                return (k1, k2)


def remove_duplicates(pdb: cfr.ProxyDatabase, test_identity=False, correlation_threshold=0.999):
    """
    Remove duplicate proxies. Can be based on identity of timeseries or correlation threshold. Timeseries are aligned
    for correlation computation and must have an overlap of 10 or more samples. Proxies cannot be duplicates if they are
    of different type, more than 1000 km apart or their temporal resolution is different.

    If duplicates are found, longer timeseries are preferred; if they are the same length, Pages2k is preferred.

    Args:
        pdb: proxy database in which to find duplicates
        test_identity: whether the identity or correlation criterion should be used
        correlation_threshold: threshold for correlation if test_identity=False (default: 0.999)

    Returns:
        proxy database with one of each duplicate pair removed
    """
    duplicates: list[tuple[str, str]] = []
    records = list(pdb.records.items())

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_is_duplicate, k1, v1, k2, v2, test_identity, correlation_threshold)
            for i, (k1, v1) in enumerate(records)
            for k2, v2 in records[i + 1 :]
        ]
        for future in tqdm(futures):
            result = future.result()
            if result is not None:
                duplicates.append(result)

    if duplicates:
        logger.info(f"Found {len(duplicates)} duplicates:")
        for pid1, pid2 in duplicates:
            logger.info(f"  {pid1} == {pid2}")

    proxies_to_remove = []
    for pid1, pid2 in duplicates:
        p1_len = len(pdb.records[pid1].time)
        p2_len = len(pdb.records[pid2].time)
        if p1_len == p2_len:
            # Keep pages2k version, remove ch2k version
            if pid1.startswith("pages2k"):
                proxies_to_remove.append(pdb.records[pid2])
            else:
                proxies_to_remove.append(pdb.records[pid1])
        else:
            # Keep the one with the longer time series
            if p1_len > p2_len:
                proxies_to_remove.append(pdb.records[pid2])
            else:
                proxies_to_remove.append(pdb.records[pid1])

    # Manually determined this is a duplicate of noaa_Palmyra2020d18O
    proxies_to_remove.append(pdb.records["pages2k_Ocn_103"])

    return pdb - proxies_to_remove


if __name__ == "__main__":
    logger.info("Loading Pages2k")
    pdb_pages2k = load_pages2k()
    logger.info("Loading CoralHydro2k")
    pdb_ch2k = load_ch2k()
    logger.info("Loading NOAA proxies")
    pdb_noaa = sum(
        [
            load_noaa_proxy(f)
            for f in (get_base_path() / "datasets" / "proxies" / "NOAA").glob("*.txt")
        ],
        cfr.ProxyDatabase({}),
    )

    logger.info("Combining and removing duplicates")
    pdb_combined = remove_duplicates(pdb_pages2k + pdb_ch2k + pdb_noaa)

    output_path = get_base_path() / "datasets" / "proxies"
    logger.info(f"Saving proxy database to {output_path}")
    pickle.dump(pdb_pages2k, (output_path / "pages2k.pkl").open("wb"))
    pickle.dump(pdb_ch2k, (output_path / "ch2k.pkl").open("wb"))
    pickle.dump(pdb_noaa, (output_path / "noaa.pkl").open("wb"))
    pickle.dump(pdb_combined, (output_path / "combined.pkl").open("wb"))

from __future__ import annotations

import pickle

import cfr
import pandas as pd
from cfr import ProxyDatabase
from pylipd.lipd import LiPD

from lmrecon.logger import get_logger
from lmrecon.util import get_base_path

logger = get_logger(__name__)


def load_pages2k():
    lipd = LiPD()
    lipd.load_from_dir(str(get_base_path() / "datasets" / "proxies" / "Pages2k"), parallel=True)

    df = lipd.get_timeseries(lipd.get_all_dataset_names(), to_dataframe=True)[1]

    # Match CFR ptypes (https://github.com/fzhu2e/cfr/blob/8598c8ef2dcc64d490e19886027670b5b32e22e7/cfr/proxy.py#L48)
    df["paleoData_proxy"] = df["paleoData_proxy"].replace(
        {
            "ring width": "TRW",
            "maximum latewood density": "MXD",
            "historical": "historic",
            "accumulation rate": "sed accumulation",
            "chrysophyte assemblage": "chrysophyte",
            "multiproxy": "hybrid",
            "ice melt": "melt",
            "d18o": "d18O",
        }
    )
    df["archiveType"] = df["archiveType"].replace(
        {
            "Wood": "tree",
            "Mollusk shell": "bivalve",
        }
    )

    df = df[df["paleoData_pages2kID"].notnull()]
    df = df[~df["paleoData_pages2kID"].str.startswith("X")]
    df["pid"] = "pages2k_" + df["paleoData_pages2kID"]

    # Should contain 692 records
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
        proxy_type_column="paleoData_variableName",
        pid_column="pid",
    )
    # Ensure we added all proxies
    assert len(pdb.records) == len(df)

    # Should contain 233 records
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
    data = pd.read_csv(file, sep=r"\s+", comment="#")
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
    pdb_combined = (
        (
            pdb_pages2k
            # Do not use corals from Pages2k, avoids duplication issues
            - pdb_pages2k.filter("ptype", "coral.d18O")
            - pdb_pages2k.filter("ptype", "coral.SrCa")
        )
        + pdb_ch2k
        + pdb_noaa
    )
    # Remove Cobb 2003 records (CO03PAL*) in favor of Dee 2020 record (Palmyra2020d18O)
    # Dee 2020 contains Cobb 2003
    # Pages2k also contains Cobb2003 (Ocn_103) but has already been removed
    proxies_to_remove = set()
    for pid, proxy in pdb_combined.records.items():
        if pid.startswith("ch2k_CO03PAL"):
            proxies_to_remove.add(proxy)
    pdb_combined = pdb_combined - proxies_to_remove

    output_path = get_base_path() / "datasets" / "proxies"
    logger.info(f"Saving proxy database to {output_path}")
    pickle.dump(pdb_pages2k, (output_path / "pages2k.pkl").open("wb"))
    pickle.dump(pdb_ch2k, (output_path / "ch2k.pkl").open("wb"))
    pickle.dump(pdb_noaa, (output_path / "noaa.pkl").open("wb"))
    pickle.dump(pdb_combined, (output_path / "combined.pkl").open("wb"))

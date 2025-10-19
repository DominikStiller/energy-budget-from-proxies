from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lmrecon.lim import LIM
from lmrecon.mapper import PhysicalSpaceForecastSpaceMapper
from lmrecon.util import get_data_path

if TYPE_CHECKING:
    import cfr


def print_obs_info(id, print_mapper=True):
    with (get_data_path() / "obs" / id / "metadata.json").open() as f:
        metadata = json.load(f)
    pdb: cfr.ProxyDatabase = pickle.load((get_data_path() / "obs" / id / "pdb.pkl").open("rb"))
    pickle.load((get_data_path() / "obs" / id / "psms.pkl").open("rb"))

    print(f"=== Observations ({id}) ===")
    print("n:", len(pdb.records))
    print("pdb path:", metadata["pdb_path"])
    print("minimum calibration overlap:", metadata["minimum_calibration_overlap"])
    print("minimum corr threshold:", metadata["minimum_corr_threshold"])
    print("maximum annual error acor threshold:", metadata["maximum_annual_error_acor_threshold"])

    if print_mapper:
        print()
        print_mapper_info(metadata["mapper_id"])


def print_pseudoobs_info(id, print_mapper=True):
    with (get_data_path() / "pseudoobs" / id / "metadata.json").open() as f:
        metadata = json.load(f)
    pdb: cfr.ProxyDatabase = pickle.load(
        (get_data_path() / "pseudoobs" / id / "pdb.pkl").open("rb")
    )
    pickle.load((get_data_path() / "pseudoobs" / id / "psms.pkl").open("rb"))

    print(f"=== Pseudo Observations ({id}) ===")
    print("n:", len(pdb.records))
    print("true dataset:", metadata["true_dataset"])
    print("type:", metadata["type"])

    print()
    print_obs_info(metadata["calibrated_obs_dataset"], print_mapper=False)


def print_mapper_info(id):
    with (get_data_path() / "mapper" / id / "metadata.json").open() as f:
        metadata = json.load(f)
    mapper = PhysicalSpaceForecastSpaceMapper.load(get_data_path() / "mapper" / id / "mapper.pkl")

    print(f"=== Mapper ({id}) ===")
    print("fields:", ", ".join(mapper.fields))
    print("k:", metadata["k"])
    print("l:", metadata["l"])
    print("k_direct:", metadata["k_direct"])
    print("state dimension: ", mapper.n_reduced_state)
    print("physical dataset:", metadata["physical_dataset"])
    print("Standardize by season:", metadata["standardize_by_season"])
    print("Separate global mean:", metadata["separate_global_mean"])
    print("Retained variance:")
    for field, eof in mapper.eofs_individual.items():
        print(f" - {field}: {eof.variance_fraction_retained:.0%}")


def print_lim_info(id):
    with (get_data_path() / "lim" / id / "metadata.json").open() as f:
        metadata = json.load(f)
    lim = LIM.load(get_data_path() / "lim" / id / "lim.pkl")

    print(f"=== LIM ({id}) ===")
    print("state dimension: ", lim.Nx)
    print()
    print_mapper_info(metadata["mapper_id"])


def print_reconstruction_info(id):
    with (get_data_path() / "reconstructions" / id / "metadata.json").open() as f:
        metadata = json.load(f)

    print(f"=== Reconstruction ({id}) ===")
    print(f"Years: {metadata['year_start']}-{metadata['year_end']}")
    print("Obs dataset:", metadata["obs_dataset"])
    print("Initial prior dataset:", metadata["initial_prior_dataset"])
    print("Ensemble size:", metadata["n_ens"])
    print("Assimilated fraction:", metadata["assimilated_fraction"])
    print("Seed offset:", metadata["seed_offset"])
    print("Window mode:", metadata["window_mode"])
    print("Removed proxies:", metadata["removed_proxies"])

    print()
    obs_dataset = metadata["obs_dataset"]
    if obs_dataset.startswith("pseudoobs/"):
        print_pseudoobs_info(obs_dataset.split("/")[-1], print_mapper=False)
    else:
        print_obs_info(obs_dataset.split("/")[-1], print_mapper=False)

    print()
    if "lim_id" in metadata:
        print_lim_info(metadata["lim_id"])
    else:
        print_mapper_info(metadata["mapper_id"])


if __name__ == "__main__":
    path = Path(sys.argv[1])
    if "reconstructions" in path.parts:
        id = path.parts[path.parts.index("reconstructions") + 1]
        print_reconstruction_info(id)
    elif "lim" in path.parts:
        id = path.parts[path.parts.index("lim") + 1]
        print_lim_info(id)
    elif "obs" in path.parts:
        id = path.parts[path.parts.index("obs") + 1]
        print_obs_info(id)
    elif "mapper" in path.parts:
        id = path.parts[path.parts.index("mapper") + 1]
        print_mapper_info(id)

import os
import logging
from datetime import datetime, timedelta
import shutil

import numpy as np
import pandas as pd

from autumn import db
from autumn.region import Region
from autumn.db.database import Database
from autumn.tool_kit.params import load_targets
from utils.timer import Timer
from utils.parallel import run_parallel_tasks
from utils.s3 import list_s3, upload_s3, download_from_s3
from settings import REMOTE_BASE_DIR


DHHS_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "dhhs")
DHHS_POWERBI_DATA_DIR = os.path.join(DHHS_DATA_DIR, "powerbi")
DHHS_FULL_RUN_DATA_DIR = os.path.join(DHHS_DATA_DIR, "full")

DATESTAMP = datetime.now().isoformat().split(".")[0].replace(":", "-")
BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)
ENSEMBLE_OUTPUT = "notifications_at_sympt_onset"
OUTPUTS = [
    "incidence",
    "notifications",
    "infection_deaths",
    "new_icu_admissions",
    "hospital_occupancy",
    "new_icu_admissions",
    "icu_occupancy",
]
DHHS_NUM_CHOSEN = 800
ENSEMBLE_NUM_CHOSEN = 1000

logger = logging.getLogger(__name__)


def dhhs_task(commit: str, quiet: bool):
    with Timer(f"Fetching file info from AWS S3 for commit {commit}"):
        vic_full_run_db_keys = get_vic_full_run_dbs_for_commit(commit)
        vic_powerbi_db_keys = get_vic_powerbi_dbs_for_commit(commit)

    # Set up directories for output data.
    with Timer(f"Creating directories"):
        for dirpath in [DHHS_DATA_DIR, DHHS_POWERBI_DATA_DIR, DHHS_FULL_RUN_DATA_DIR]:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)

            os.makedirs(dirpath)

    # Download all PowerBI databases.
    with Timer(f"Downloading PowerBI databases"):
        args_list = [(src_key, quiet) for src_key in vic_powerbi_db_keys]
        run_parallel_tasks(download_powerbi_db_from_s3, args_list)

    # Download all full model run databases.
    for region, keys in vic_full_run_db_keys.items():
        with Timer(f"Downloading full model run databases for {region}"):
            args_list = [(src_key, region, quiet) for src_key in keys]
            run_parallel_tasks(download_full_db_from_s3, args_list)

    # Build Victorian regional CSV file
    filename = f"vic-forecast-{commit}-{DATESTAMP}.csv"
    csv_path = os.path.join(DHHS_DATA_DIR, filename)
    with Timer(f"Building Victorian regional CSV file."):
        for db_name in os.listdir(DHHS_POWERBI_DATA_DIR):
            db_path = os.path.join(DHHS_POWERBI_DATA_DIR, db_name)
            powerbi_db = Database(db_path)
            df = powerbi_db.query("uncertainty", conditions={"scenario": 0})
            df.drop(columns=["scenario"], inplace=True)
            df.time = df.time.apply(lambda days: BASE_DATETIME + timedelta(days=days))
            df["region"] = "_".join(db_name.split("-")[2:-2]).upper()
            df = df[["region", "type", "time", "quantile", "value"]]
            if os.path.exists(csv_path):
                df.to_csv(csv_path, float_format="%.8f", mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, float_format="%.8f", mode="w", index=False)

    # Build the combined Victorian CSV file.
    with Timer(f"Building Victorian whole-state CSV file."):
        targets = load_targets("covid_19", "hume")
        targets = {k: {**v, "times": [], "values": []} for k, v in targets.items()}
        outputs = [t["output_key"] for t in targets.values()]

        # Get sample runs from all chains
        start_t = 140
        end_t = 400
        times = np.linspace(start_t, end_t, end_t - start_t + 1)
        samples = []
        for _ in range(DHHS_NUM_CHOSEN):
            sample = {}
            sample["weights"] = np.zeros(len(times))
            for o in outputs:
                sample[o] = np.zeros(len(times))

            samples.append(sample)

        for region in Region.VICTORIA_SUBREGIONS:
            if region == Region.VICTORIA:
                continue

            region_dir = os.path.join(DHHS_FULL_RUN_DATA_DIR, region)
            mcmc_tables = db.load.append_tables(db.load.load_mcmc_tables(region_dir))
            chosen_runs = db.process.sample_runs(mcmc_tables, DHHS_NUM_CHOSEN)
            mcmc_tables = mcmc_tables.set_index(["chain", "run"])
            do_tables = db.load.load_derived_output_tables(region_dir)
            do_tables = db.load.append_tables(do_tables).set_index(["chain", "run"])

            for idx, (chain, run) in enumerate(chosen_runs):
                sample = samples[idx]
                sample["weights"] += mcmc_tables.loc[(chain, run), "weight"]
                run_start_t = do_tables.loc[(chain, run), "times"].iloc[0]
                t_offset = int(run_start_t - start_t)
                for o in outputs:
                    sample[o][t_offset:] += do_tables.loc[(chain, run), o]

        columns = ["scenario", "times", "weight", *outputs]
        data = {
            "scenario": np.concatenate([np.zeros(len(times)) for _ in range(DHHS_NUM_CHOSEN)]),
            "times": np.concatenate([times for _ in range(DHHS_NUM_CHOSEN)]),
            "weight": np.concatenate([s["weights"] for s in samples]),
        }
        for o in outputs:
            data[o] = np.concatenate([s[o] for s in samples])

        df = pd.DataFrame(data=data, columns=columns)
        uncertainty_df = db.uncertainty._calculate_mcmc_uncertainty(df, targets)

        # Add Victorian uncertainty to the existing CSV
        baseline_mask = uncertainty_df["scenario"] == 0
        uncertainty_df = uncertainty_df[baseline_mask].copy()
        uncertainty_df.drop(columns=["scenario"], inplace=True)
        uncertainty_df.time = uncertainty_df.time.apply(
            lambda days: BASE_DATETIME + timedelta(days=days)
        )
        uncertainty_df["region"] = "VICTORIA"
        uncertainty_df = uncertainty_df[["region", "type", "time", "quantile", "value"]]
        uncertainty_df.to_csv(csv_path, float_format="%.8f", mode="a", header=False, index=False)

        # Upload the CSV
        s3_dest_key = f"dhhs/{filename}"
        upload_s3(csv_path, s3_dest_key)

    # Build the ensemble forecast CSV file.
    with Timer(f"Building Victorian ensemble forecast CSV file."):
        filename = f"vic-ensemble-{commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DATA_DIR, filename)

        # Get sample runs from all chains
        start_t = (datetime.now() - BASE_DATETIME).days  # Now
        end_t = min(start_t + 7 * 6 + 1, 400)  # 6 weeks from now
        times = np.linspace(start_t, end_t - 1, end_t - start_t, dtype=np.int64)
        samples = [np.zeros(end_t - start_t) for _ in range(ENSEMBLE_NUM_CHOSEN)]
        for region in Region.VICTORIA_SUBREGIONS:
            if region == Region.VICTORIA:
                continue

            region_dir = os.path.join(DHHS_FULL_RUN_DATA_DIR, region)
            mcmc_tables = db.load.append_tables(db.load.load_mcmc_tables(region_dir))
            chosen_runs = db.process.sample_runs(mcmc_tables, ENSEMBLE_NUM_CHOSEN)
            do_tables = db.load.load_derived_output_tables(region_dir)
            do_tables = db.load.append_tables(do_tables).set_index(["chain", "run"])
            for idx, (chain, run) in enumerate(chosen_runs):
                run_start_t = do_tables.loc[(chain, run), "times"].iloc[0]
                start_idx = int(start_t - run_start_t)
                end_idx = int(end_t - run_start_t)
                df = do_tables.loc[(chain, run)].iloc[start_idx:end_idx]
                is_times_equal = (df["times"].to_numpy() == times).all()
                assert is_times_equal, "All samples must have correct time range"
                samples[idx] += df[ENSEMBLE_OUTPUT].to_numpy()

        columns = ["run", "times", ENSEMBLE_OUTPUT]
        data = {
            "run": np.concatenate(
                [idx * np.ones(len(times), dtype=np.int64) for idx in range(ENSEMBLE_NUM_CHOSEN)]
            ),
            "times": np.concatenate([times for _ in range(ENSEMBLE_NUM_CHOSEN)]),
            ENSEMBLE_OUTPUT: np.concatenate(samples),
        }
        df = pd.DataFrame(data=data, columns=columns)
        df["times"] = df["times"].apply(lambda days: BASE_DATETIME + timedelta(days=days))
        df.to_csv(csv_path, float_format="%.8f", index=False)

        # Upload the CSV
        s3_dest_key = f"ensemble/{filename}"
        upload_s3(csv_path, s3_dest_key)


def download_full_db_from_s3(src_key, region, quiet):
    chain_path = src_key.split("/full_model_runs/")[-1]
    dest_path = os.path.join(DHHS_FULL_RUN_DATA_DIR, region, chain_path)
    if os.path.exists(dest_path):
        os.remove(dest_path)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    download_from_s3(src_key, dest_path, quiet)


def download_powerbi_db_from_s3(src_key, quiet):
    filename = src_key.split("/")[-1]
    dest_path = os.path.join(DHHS_POWERBI_DATA_DIR, filename)
    if os.path.exists(dest_path):
        os.remove(dest_path)

    download_from_s3(src_key, dest_path, quiet)


def get_vic_full_run_dbs_for_commit(commit: str):
    keys = {}
    for region in Region.VICTORIA_SUBREGIONS:
        if region == Region.VICTORIA:
            continue

        key_prefix = os.path.join("covid_19", region)
        region_db_keys = list_s3(key_prefix, key_suffix=".feather")
        filter_key = lambda k: (
            commit in k and "full_model_runs" in k and ("derived_outputs" in k or "mcmc_run" in k)
        )
        region_db_keys = [k for k in region_db_keys if filter_key(k)]
        msg = f"There should exactly one set of full model run databases for {region} with commit {commit}: {region_db_keys}"
        filenames = ["/".join(k.split("/")[-2:]) for k in region_db_keys]
        assert len(filenames) == len(set(filenames)), msg
        keys[region] = region_db_keys

    return keys


def get_vic_powerbi_dbs_for_commit(commit: str):
    keys = []
    for region in Region.VICTORIA_SUBREGIONS:
        if region == Region.VICTORIA:
            continue

        prefix = f"covid_19/{region}"
        region_db_keys = list_s3(key_prefix=prefix, key_suffix=".db")
        region_db_keys = [k for k in region_db_keys if commit in k and "powerbi" in k]
        msg = f"There should exactly one PowerBI database for {region} with commit {commit}: {region_db_keys}"
        assert len(region_db_keys) == 1, msg
        keys.append(region_db_keys[0])

    return keys

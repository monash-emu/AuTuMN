import os
import logging
from datetime import datetime, timedelta

import luigi
import numpy as np

from autumn import db
from autumn.constants import Region
from autumn.db.database import Database
from autumn.tool_kit.params import load_targets
import pandas as pd

from . import utils
from . import settings

logger = logging.getLogger(__name__)


OUTPUTS = [
    "incidence",
    "notifications",
    "infection_deaths",
    "new_icu_admissions",
    "hospital_occupancy",
    "new_icu_admissions",
    "icu_occupancy",
]


DHHS_DIR = os.path.join(settings.BASE_DIR, "data", "outputs", "dhhs")
DATESTAMP = datetime.now().isoformat().split(".")[0].replace(":", "-")
BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


class RunDHHS(luigi.Task):
    """DHHS post processing master task"""

    commit = luigi.Parameter()

    def requires(self):
        return [
            BuildFinalCSVTask(commit=self.commit),
            BuildEnsembleTask(commit=self.commit),
        ]


class BuildEnsembleTask(utils.BaseTask):
    commit = luigi.Parameter()

    def output(self):
        filename = f"vic-ensemble-{self.commit}-{DATESTAMP}.csv"
        s3_uri = os.path.join(f"s3://{settings.S3_BUCKET}", f"ensemble/{filename}")
        return utils.S3Target(s3_uri, client=utils.luigi_s3_client)

    def requires(self):
        dbs = get_vic_full_run_dbs_for_commit(self.commit)
        downloads = []
        for region, s3_keys in dbs.items():
            for s3_key in s3_keys:
                task = DownloadFullModelRunTask(s3_key=s3_key, region=region)
                downloads.append(task)

        return downloads

    def safe_run(self):
        """
        Build predections for non-vic ensemble reporting
        """
        filename = f"vic-ensemble-{self.commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DIR, filename)
        OUTPUT = "notifications_at_sympt_onset"

        # Get sample runs from all chains
        num_chosen = 2000
        start_t = (datetime.now() - BASE_DATETIME).days  # Now
        end_t = min(start_t + 7 * 6 + 1, 365)  # 6 weeks from now
        times = np.linspace(start_t, end_t - 1, end_t - start_t, dtype=np.int64)
        samples = [np.zeros(end_t - start_t) for _ in range(num_chosen)]
        for region in Region.VICTORIA_SUBREGIONS:
            if region == Region.VICTORIA:
                continue

            region_dir = os.path.join(DHHS_DIR, "full", region)
            mcmc_tables = db.process.append_tables(db.load.load_mcmc_tables(region_dir))
            chosen_runs = db.process.sample_runs(mcmc_tables, num_chosen)
            do_tables = db.load.load_derived_output_tables(region_dir)
            do_tables = db.process.append_tables(do_tables).set_index(["chain", "run"])
            for idx, (chain, run) in enumerate(chosen_runs):
                run_start_t = do_tables.loc[(chain, run), "times"].iloc[0]
                start_idx = int(start_t - run_start_t)
                end_idx = int(end_t - run_start_t)
                df = do_tables.loc[(chain, run)].iloc[start_idx:end_idx]
                is_times_equal = (df["times"].to_numpy() == times).all()
                assert is_times_equal, "All samples must have correct time range"
                samples[idx] += df[OUTPUT].to_numpy()

        columns = ["run", "times", OUTPUT]
        data = {
            "run": np.concatenate(
                [idx * np.ones(len(times), dtype=np.int64) for idx in range(num_chosen)]
            ),
            "times": np.concatenate([times for _ in range(num_chosen)]),
            OUTPUT: np.concatenate(samples),
        }
        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(csv_path, index=False)

        # Upload the CSV
        s3_dest_key = f"ensemble/{filename}"
        utils.upload_s3(csv_path, s3_dest_key)


class BuildFinalCSVTask(utils.BaseTask):
    commit = luigi.Parameter()

    def safe_run(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DIR, filename)

        targets = load_targets("covid_19", "hume")
        targets = {k: {**v, "times": [], "values": []} for k, v in targets.items()}
        outputs = [t["output_key"] for t in targets.values()]

        # Get sample runs from all chains
        num_chosen = 800
        start_t = 140
        end_t = 365
        times = np.linspace(start_t, end_t, end_t - start_t + 1)
        samples = []
        for _ in range(num_chosen):
            sample = {}
            sample["weights"] = np.zeros(len(times))
            for o in outputs:
                sample[o] = np.zeros(len(times))

            samples.append(sample)

        for region in Region.VICTORIA_SUBREGIONS:
            if region == Region.VICTORIA:
                continue

            region_dir = os.path.join(DHHS_DIR, "full", region)
            mcmc_tables = db.process.append_tables(db.load.load_mcmc_tables(region_dir))
            chosen_runs = db.process.sample_runs(mcmc_tables, num_chosen)
            mcmc_tables = mcmc_tables.set_index(["chain", "run"])
            do_tables = db.load.load_derived_output_tables(region_dir)
            do_tables = db.process.append_tables(do_tables).set_index(["chain", "run"])

            for idx, (chain, run) in enumerate(chosen_runs):
                mcmc_row = mcmc_tables.loc[(chain, run), :]
                sample = samples[idx]
                sample["weights"] += mcmc_tables.loc[(chain, run), "weight"]

                run_start_t = do_tables.loc[(chain, run), "times"].iloc[0]
                t_offset = int(run_start_t - start_t)
                for o in outputs:
                    sample[o][t_offset:] += do_tables.loc[(chain, run), o]

        columns = ["scenario", "times", "weight", *outputs]
        data = {
            "scenario": np.concatenate([np.zeros(len(times)) for _ in range(num_chosen)]),
            "times": np.concatenate([times for _ in range(num_chosen)]),
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
        uncertainty_df.to_csv(csv_path, mode="a", header=False, index=False)

        # Upload the CSV
        s3_dest_key = f"dhhs/{filename}"
        # utils.upload_s3(csv_path, s3_dest_key)

    def output(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        s3_uri = os.path.join(f"s3://{settings.S3_BUCKET}", f"dhhs/{filename}")
        return utils.S3Target(s3_uri, client=utils.luigi_s3_client)

    def requires(self):
        dbs = get_vic_full_run_dbs_for_commit(self.commit)
        downloads = []
        for region, s3_keys in dbs.items():
            for s3_key in s3_keys:
                task = DownloadFullModelRunTask(s3_key=s3_key, region=region)
                downloads.append(task)

        return [BuildRegionCSVTask(commit=self.commit), *downloads]


class BuildRegionCSVTask(utils.BaseTask):

    commit = luigi.Parameter()

    def safe_run(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DIR, filename)
        powerbi_path = os.path.join(DHHS_DIR, "powerbi")
        for db_name in os.listdir(powerbi_path):
            db_path = os.path.join(powerbi_path, db_name)
            db = Database(db_path)
            df = db.query("uncertainty", conditions=["scenario=0"])
            df.drop(columns=["scenario"], inplace=True)
            df.time = df.time.apply(lambda days: BASE_DATETIME + timedelta(days=days))
            df["region"] = "_".join(db_name.split("-")[1:-2]).upper()
            df = df[["region", "type", "time", "quantile", "value"]]
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(csv_path, mode="w", index=False)

    def output(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DIR, filename)
        return luigi.LocalTarget(csv_path)

    def requires(self):
        s3_keys = get_vic_powerbi_dbs_for_commit(self.commit)
        return [DownloadPowerBITask(s3_key=s3_key) for s3_key in s3_keys]


class DownloadFullModelRunTask(utils.BaseTask):

    s3_key = luigi.Parameter()
    region = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.get_dest_path())

    def safe_run(self):
        dest_path = self.get_dest_path()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        utils.download_s3(self.s3_key, dest_path)

    def get_dest_path(self):
        return os.path.join(DHHS_DIR, "full", self.region, self.filename)

    @property
    def filename(self):
        return self.s3_key.split("/")[-1]


class DownloadPowerBITask(utils.BaseTask):

    s3_key = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.get_dest_path())

    def safe_run(self):
        dest_path = self.get_dest_path()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        utils.download_s3(self.s3_key, dest_path)

    def get_dest_path(self):
        return os.path.join(DHHS_DIR, "powerbi", self.filename)

    @property
    def filename(self):
        return self.s3_key.split("/")[-1]


def get_vic_full_run_dbs_for_commit(commit: str):
    keys = {}
    for region in Region.VICTORIA_SUBREGIONS:
        if region == Region.VICTORIA:
            continue

        prefix = f"covid_19/{region}"
        region_db_keys = utils.list_s3(key_prefix=prefix, key_suffix=".db")
        region_db_keys = [k for k in region_db_keys if commit in k and "mcmc_chain_full_run" in k]

        msg = f"There should exactly one set of full model run databases for {region} with commit {commit}: {region_db_keys}"
        filenames = [k.split("/")[-1] for k in region_db_keys]
        assert len(filenames) == len(set(filenames)), msg
        keys[region] = region_db_keys

    return keys


def get_vic_powerbi_dbs_for_commit(commit: str):
    keys = []
    for region in Region.VICTORIA_SUBREGIONS:
        if region == Region.VICTORIA:
            continue

        prefix = f"covid_19/{region}"
        region_db_keys = utils.list_s3(key_prefix=prefix, key_suffix=".db")
        region_db_keys = [k for k in region_db_keys if commit in k and "powerbi" in k]

        msg = f"There should exactly one PowerBI database for {region} with commit {commit}: {region_db_keys}"
        assert len(region_db_keys) == 1, msg
        keys.append(region_db_keys[0])

    return keys

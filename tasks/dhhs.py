import os
import logging
from datetime import datetime, timedelta

import luigi
import numpy as np

from autumn import db
from autumn.constants import Region
from autumn.db.database import Database
from autumn.tool_kit.params import load_targets

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
        return BuildFinalCSVTask(commit=self.commit)


class BuildFinalCSVTask(utils.BaseTask):
    commit = luigi.Parameter()

    def safe_run(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DIR, filename)

        targets = load_targets("covid_19", "hume")
        targets = {k: {**v, "times": [], "values": []} for k, v in targets.items()}
        outputs = [t["output_key"] for t in targets.values()]

        # Get sample runs from all chains
        mcmc_df = None
        do_df = None
        num_chosen_per_chain = 120
        for region in Region.VICTORIA_SUBREGIONS:
            if region == Region.VICTORIA:
                continue

            region_dir = os.path.join(DHHS_DIR, "full", region)
            mcmc_tables = db.process.append_tables(db.load.load_mcmc_tables(region_dir))
            do_tables = db.process.append_tables(db.load.load_derived_output_tables(region_dir))
            num_chains = mcmc_tables.chain.max() + 1
            num_chosen = num_chosen_per_chain * num_chains
            chosen_runs = db.process.sample_runs(mcmc_tables, num_chosen)

            mcmc_mask = None
            for chain, run in chosen_runs:
                mask = (mcmc_tables["chain"] == chain) & (mcmc_tables["run"] == run)
                mcmc_mask = mask if mcmc_mask is None else (mcmc_mask | mask)

            do_mask = None
            for chain, run in chosen_runs:
                mask = (do_tables["chain"] == chain) & (do_tables["run"] == run)
                do_mask = mask if do_mask is None else (do_mask | mask)

            mcmc_chosen = mcmc_tables[mcmc_mask].copy()
            do_chosen = do_tables[do_mask].copy()
            assert len(mcmc_chosen) == num_chosen

            # Aggregate weights over regions
            if mcmc_df is None:
                mcmc_df = mcmc_chosen
            else:
                mcmc_df["weight"] = mcmc_df["weight"].to_numpy() + mcmc_chosen["weight"].to_numpy()

            # Aggregate outputs over regions
            if do_df is None:
                do_df = do_chosen
            else:
                common_times = np.intersect1d(do_df["times"], do_chosen["times"])
                min_t, max_t = common_times.min(), common_times.max()
                do_df_mask = (do_df["times"] >= min_t) & (do_df["times"] <= max_t)
                do_chosen_mask = (do_chosen["times"] >= min_t) & (do_chosen["times"] <= max_t)
                for output in outputs:
                    output_sum = (
                        do_df[do_df_mask][output].to_numpy()
                        + do_chosen[do_chosen_mask][output].to_numpy()
                    )
                    do_df[do_df_mask][output] = output_sum

        uncertainty_df = db.uncertainty.calculate_mcmc_uncertainty(mcmc_df, do_df, targets)

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
        # s3_dest_key = f"dhhs/{filename}"
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

        region_db_keys = utils.list_s3(key_prefix=region, key_suffix=".db")
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

        region_db_keys = utils.list_s3(key_prefix=region, key_suffix=".db")
        region_db_keys = [k for k in region_db_keys if commit in k and "powerbi" in k]

        msg = f"There should exactly one PowerBI database for {region} with commit {commit}: {region_db_keys}"
        assert len(region_db_keys) == 1, msg
        keys.append(region_db_keys[0])

    return keys

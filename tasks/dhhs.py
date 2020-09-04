import os
import logging
from datetime import datetime, timedelta

import luigi

from autumn.constants import Region
from autumn.db.database import Database

from . import utils
from . import settings

logger = logging.getLogger(__name__)


OUTPUTS = [
    "incidence",
    "notifications",
    "total_infection_deaths",
    "new_icu_admissions",
    "hospital_occupancy",
    "new_icu_admissions",
    "icu_occupancy",
]

"""
1. Get run ids for commit, validate them.
2. Read in PowerBI database for each region
3. Collate uncertainties into a single CSV for all OUTPUTS
4. Put results CSV somewhere
5. Update website build
"""
DHHS_DIR = os.path.join(settings.BASE_DIR, "data", "outputs", "dhhs")
DATESTAMP = datetime.now().isoformat().split(".")[0].replace(":", "-")
BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


class RunDHHS(luigi.Task):
    """DHHS post processing master task"""

    commit = luigi.Parameter()

    def requires(self):
        return BuildRegionCSVTask(commit=self.commit)


class BuildFinalCSVTask(utils.BaseTask):
    commit = luigi.Parameter()

    def safe_run(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        csv_path = os.path.join(DHHS_DIR, filename)
        s3_dest_key = f"/dhhs/{filename}"

        # Upload the CSV
        utils.upload_s3(csv_path, s3_dest_key)

    def output(self):
        filename = f"vic-forecast-{self.commit}-{DATESTAMP}.csv"
        s3_uri = os.path.join(f"s3://{settings.S3_BUCKET}", f"/dhhs/{filename}")
        return utils.S3Target(s3_uri, client=utils.luigi_s3_client)

    def requires(self):
        # dbs = get_vic_full_run_dbs_for_commit(self.commit)
        # downloads = [DownloadFullModelRunTask(s3_key=k, region=r) for r, k in dbs.items()]
        downloads = []
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
            df = db.query("uncertainty", conditions=["Scenario='S_0'"])
            df.drop(columns=["Scenario"], inplace=True)
            df.time = df.time.apply(lambda days: BASE_DATETIME + timedelta(days=days))
            df["region"] = "_".join(db_name.split("-")[1:-2]).upper()
            df = df[["region", "type", "time", "quantile", "value"]]
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", header=False)
            else:
                df.to_csv(csv_path, mode="w")

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
        s3_uri = os.path.join(f"s3://{settings.S3_BUCKET}", self.s3_key)
        download_s3(s3_uri, dest_path)

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
        s3_uri = os.path.join(f"s3://{settings.S3_BUCKET}", self.s3_key)
        download_s3(s3_uri, dest_path)

    def get_dest_path(self):
        return os.path.join(DHHS_DIR, "powerbi", self.filename)

    @property
    def filename(self):
        return self.s3_key.split("/")[-1]


def get_vic_full_run_dbs_for_commit(commit: str):
    keys = {}
    for region in Region.VICTORIA_SUBREGIONS:
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
        region_db_keys = utils.list_s3(key_prefix=region, key_suffix=".db")
        region_db_keys = [k for k in region_db_keys if commit in k and "powerbi" in k]
        msg = f"There should exactly one PowerBI database for {region} with commit {commit}: {region_db_keys}"
        assert len(region_db_keys) == 1, msg
        keys.append(region_db_keys[0])

    return keys

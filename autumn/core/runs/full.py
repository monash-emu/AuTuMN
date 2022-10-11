from pathlib import PurePosixPath
import re

from autumn.core import db


class ManagedFullRun:
    def __init__(self, manager):
        self._manager = manager
        self.local_path = self._manager.local_path / "data/full_model_runs"
        self.remote_path = PurePosixPath(self._manager.run_id) / "data/full_model_runs"
        self._collated_db = None

    def download_outputs(self):
        fmatchstr = "*.feather"
        output_str = f"{self._manager.run_id}/data/full_model_runs/.*/{fmatchstr}"
        for f in self._manager.remote.list_contents():
            m = re.match(output_str, f)
            if m:
                print(f"Downloading {f}")
                self._manager.remote.download(f)

    def _get_outputs(self, table_name):
        db_path = self.local_path / "full_run_collated.db"
        if not db_path.exists():
            self._collate(True)
        if self._collated_db is None:
            self._collated_db = db.get_database(str(db_path))
        return self._collated_db.query(table_name)

    def get_derived_outputs(self):
        return self._get_outputs("derived_outputs")

    def get_mcmc_runs(self):
        return self._get_outputs("mcmc_run")

    def get_mcmc_params(self):
        return self._get_outputs("mcmc_params")

    def _collate(self, auto_download=True):
        try:
            database_paths = db.load.find_db_paths(str(self.local_path))
        except:
            if auto_download:
                self.download_outputs()
                database_paths = db.load.find_db_paths(str(self.local_path))
            else:
                raise FileNotFoundError(self.local_path, "Try downloading data")
        collated_db_path = str(self.local_path / "full_run_collated.db")
        db.process.collate_databases(
            database_paths, collated_db_path, tables=["derived_outputs", "mcmc_run", "mcmc_params"]
        )

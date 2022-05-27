import re
import yaml

from autumn.coreport db

from autumn.corens.utils import collate_columns_to_urun

class ManagedCalibrationRun:
    def __init__(self, manager):
        self._manager = manager
        self.data_path = self._manager.local_path / 'data/calibration_outputs'
        self._collated_db = None
        
    def get_mcmc_runs(self, raw=False, auto_download=True):
        if self._collated_db is None:
            db_path = self.data_path / 'mcmc_collated.db'
            if not db_path.exists():
                self._collate(auto_download)
            self._collated_db = db.get_database(str(db_path))

        runs = self._collated_db.query('mcmc_run')
        if not raw:
            runs = collate_columns_to_urun(runs)
            runs = runs.pivot_table(index='urun')
        return runs
        
    def get_mcmc_params(self, raw=False, auto_download=True):
        if self._collated_db is None:
            db_path = self.data_path / 'mcmc_collated.db'
            if not db_path.exists():
                self._collate(auto_download)
            self._collated_db = db.get_database(str(db_path))
            
        params = self._collated_db.query('mcmc_params')
        if not raw:
            params = collate_columns_to_urun(params,drop=True)
            params = params.pivot_table(index='urun',columns='name')
            params.columns = params.columns.droplevel()
        return params

    def get_mle_params(self, auto_download=True):
        return self._get_meta('mle-params.yml', auto_download)

    def get_priors(self, auto_download=True):
        return self._get_meta('priors-0.yml', auto_download)

    def get_params(self, auto_download=True):
        return self._get_meta('params-0.yml', auto_download)

    def get_targets(self, auto_download=True):
        return self._get_meta('targets-0.yml', auto_download)

    def _get_meta(self, path_ext, auto_download=True):
        meta_path = self.data_path / path_ext
        if not meta_path.exists():
            if auto_download:
                self.download_meta()
            else:
                raise FileNotFoundError(meta_path)
        return yaml.load(open(meta_path, 'r'), Loader=yaml.UnsafeLoader)
    
    def _collate(self, auto_download=True):
        try:
            database_paths = db.load.find_db_paths(str(self.data_path))
        except:
            if auto_download:
                self.download_mcmc()
                database_paths = db.load.find_db_paths(str(self.data_path))
            else:
                raise FileNotFoundError(self.data_path, "Try downloading data")
        collated_db_path = str(self.data_path / 'mcmc_collated.db')
        db.process.collate_databases(
                database_paths, collated_db_path, tables=["mcmc_run", "mcmc_params"]
            )
        
    def download_mcmc(self):
        mcmc_mstr = f"{self._manager.run_id}/data/calibration_outputs/.*/mcmc_.*.parquet"
        for f in self._manager.remote.list_contents():
            m = re.match(mcmc_mstr, f)
            if m:
                print(f"Downloading {f}")
                self._manager.remote.download(f)

    def download_meta(self):
        """Download all metadata files - anything that's a .yml
        """
        for f in self._manager.remote.list_contents('.yml'):
            self._manager.remote.download(f)

    def download_outputs(self, include_full_outputs=False):
        if include_full_outputs:
            fmatchstr = ".*outputs.parquet"
        else:
            fmatchstr = "derived_outputs.parquet"
        output_str = f"{self._manager.run_id}/data/calibration_outputs/.*/{fmatchstr}"
        for f in self._manager.remote.list_contents():
            m = re.match(output_str, f)
            if m:
                print(f"Downloading {f}")
                self._manager.remote.download(f)

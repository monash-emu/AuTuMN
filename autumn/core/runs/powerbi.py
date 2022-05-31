from pathlib import Path

from summer.utils import ref_times_to_dti

from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.core import db

from.utils import collate_columns_to_urun

class ManagedPowerBI:
    def __init__(self, manager):
        self._manager = manager
        self._file_str = 'powerbi-' + '-'.join(self._manager.run_id.split('/')) + '.db'
        self.local_path = self._manager.local_path / 'data/powerbi' / self._file_str
        remote_path = Path(self._manager.run_id) / 'data/powerbi' / self._file_str
        self.remote_path = remote_path.as_posix()
        
    def download(self):
        self._manager.remote.download(self.remote_path)
        
    def get_db(self, auto_download=True):
        if not self.local_path.exists():
            if auto_download:
                self.download()
            else:
                raise FileNotFoundError(self.local_path)
        return PowerBIDatabase(self.local_path)
        

class PowerBIDatabase:
    def __init__(self, db_path):
        self.db = db.get_database(str(db_path))
        self._setup()
        
    def get_uncertainty(self):
        udf = self.db.query('uncertainty')
        upt = udf.pivot_table(index='time',columns=['type','scenario','quantile'])
        upt.columns = upt.columns.droplevel()
        upt.index = ref_times_to_dti(COVID_BASE_DATETIME, upt.index)
        return upt
    
    def get_derived_outputs(self):
        dodf = self.db.query('derived_outputs')
        dodf = collate_columns_to_urun(dodf,True)
        dodf = dodf.pivot_table(index='times',columns=['scenario'])
        dodf.index = ref_times_to_dti(COVID_BASE_DATETIME, dodf.index)
        return dodf
    
    def _setup(self):
        self._setup_scenarios()
        
    def _setup_scenarios(self):
        scenario = self.db.query('scenario')
        tidx = ref_times_to_dti(COVID_BASE_DATETIME, scenario['start_time'])
        scenario = scenario.set_index('scenario')
        scenario['start_time'] = tidx
        self.scenarios = scenario
        
    def get_targets(self):
        df = self.db.query('targets').pivot_table(index='times',columns='key')
        df.columns = df.columns.droplevel()
        df.index = ref_times_to_dti(COVID_BASE_DATETIME, df.index)
        return df

from autumn.projects.sm_covid2.common_school.runner_tools import INCLUDED_COUNTRIES
import multiprocessing as mp
import sys

from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
from estival.sampling import tools as esamp
from pathlib import Path
import arviz as az


if __name__ == "__main__":

    # Retrieve country iso3 to run
    array_task_id = int(sys.argv[2])  # specific to this particular run/country
    iso3 = list(INCLUDED_COUNTRIES['all'].keys())[array_task_id - 1]
    print(f"Start job #{array_task_id}, iso3={iso3}", flush=True)

    mp.set_start_method("spawn")  # previously "forkserver"

    temp_data_path = Path.home() / "sh30/users/rragonnet/temp_data/"

    for analysis in ['main', 'increased_hh_contacts', 'no_google_mobility']:
        b_idata_path = temp_data_path / "burnt_idatas" / f"b_idata_{iso3}_{analysis}.nc"
        burnt_idata = az.from_netcdf(b_idata_path)

        bcm = get_bcm_object(iso3, analysis)
        lle = esamp.likelihood_extras_for_idata(burnt_idata, bcm)

        outpath = temp_data_path / "lle_csvs" / f"lle_{iso3}_{analysis}.csv"  
        lle.to_csv(outpath)

        print(f"Finished analysis {analysis}")
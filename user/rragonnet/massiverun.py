from autumn.projects.sm_covid2.common_school.runner_tools import (
    run_full_analysis, 
    INCLUDED_COUNTRIES,
    DEFAULT_RUN_CONFIG, 
    TEST_RUN_CONFIG
)
import multiprocessing as mp
import sys
from pathlib import Path
from time import time

ANALYSIS = 'main'  # ["main", "no_google_mobility", "increased_hh_contacts"]
RUN_CONFIG = TEST_RUN_CONFIG

if __name__ == "__main__":
    start_time = time()

    # Retrieve country iso3 to run
    array_task_id = int(sys.argv[2])  # specific to this particular run/country
    iso3 = list(INCLUDED_COUNTRIES['all'].keys())[array_task_id - 1]
    print(f"Start job #{array_task_id}, iso3={iso3}, analysis={ANALYSIS}", flush=True)

    mp.set_start_method("spawn")  # previously "forkserver"

    # create parent output directory for multi-country analysis if required
    analysis_name = "test_full_analysis_24Jan2024"
    output_root_dir = Path.home() / "sh30/users/rragonnet/outputs/"
    array_job_id = sys.argv[1]  # common to all the tasks from this array job
    analysis_output_dir = output_root_dir / f"{array_job_id}_{analysis_name}_{ANALYSIS}"
    analysis_output_dir.mkdir(exist_ok=True)

    # create country-specific output dir
    country_output_dir = analysis_output_dir / iso3
    country_output_dir.mkdir(exist_ok=True)

    _, _, _ = run_full_analysis(iso3, ANALYSIS, RUN_CONFIG, country_output_dir)
    
    print(f"Finished in {time() - start_time} seconds", flush=True)

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

ANALYSIS = 'main'
RUN_CONFIG = TEST_RUN_CONFIG

if __name__ == "__main__":
    start_time = time()

    # Retrieve country iso3 to run
    job_array_index = int(sys.argv[1])
    iso3 = list(INCLUDED_COUNTRIES['google_mobility'].keys())[job_array_index - 1]
    print(f"Start job #{job_array_index}, iso3={iso3}, analysis={ANALYSIS}", flush=True)

    # mp.set_start_method("forkserver")

    # create parent output directory for multi-country analysis if required
    analysis_name = "short_full_test_array"
    output_root_dir = Path.home() / "sh30/users/rragonnet/outputs/"
    timestamp = sys.argv[2] 
    analysis_output_dir = output_root_dir / f"{timestamp}_{analysis_name}_{ANALYSIS}"
    analysis_output_dir.mkdir(exist_ok=True)

    # create country-specific output dir
    country_output_dir = analysis_output_dir / iso3
    country_output_dir.mkdir(exist_ok=True)

    _, _, _ = run_full_analysis(iso3, ANALYSIS, RUN_CONFIG, country_output_dir)
    
    print(f"Finished in {time() - start_time} seconds", flush=True)

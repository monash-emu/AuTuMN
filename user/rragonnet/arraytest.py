# from autumn.projects.sm_covid2.common_school.runner_tools import run_full_analysis, DEFAULT_RUN_CONFIG, TEST_RUN_CONFIG
import multiprocessing as mp
import sys
# from pathlib import Path
# from time import time


if __name__ == "__main__":
    print("Start job")
    mp.set_start_method("forkserver")

    print(f"This is job #{sys.argv[1]}")

    # run_name = "long_full_test"
    # this_dir_path = Path(__file__).resolve().parent
    # out_path = this_dir_path / "outputs" / run_name
    # out_path.mkdir(exist_ok=True)

    # _, _, _ = run_full_analysis("FRA", "main", DEFAULT_RUN_CONFIG, out_path)
    
    # print(f"Finished {time() - start_time} seconds", flush=True)
    print("success!!!")
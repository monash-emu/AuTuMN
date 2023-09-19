print("start imports")
from autumn.projects.sm_covid2.common_school.runner_tools import run_full_analysis, TEST_RUN_CONFIG, DEFAULT_RUN_CONFIG
from pathlib import Path

print("finish imports")

run_name = "this_is_a_test_job"
this_dir_path = Path(__file__).resolve().parent
out_path = this_dir_path / "outputs" / run_name
out_path.mkdir(exist_ok=True)


print("Starts running the analysis")

_, _, _ = run_full_analysis(
    iso3="FRA", 
    analysis="main",
    run_config=TEST_RUN_CONFIG,
    output_folder=out_path
)

print("Analysis completed")
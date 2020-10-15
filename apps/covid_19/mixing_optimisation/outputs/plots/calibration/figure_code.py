from autumn.tool_kit.params import load_targets
from autumn import db


# --------------  Load outputs form databases
def get_calibration_outputs(region_name):
    calib_dirpath = f"../../pbi_databases/calibration/{region_name}/"
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)
    return mcmc_tables, mcmc_params


def get_targets(region_name):
    return load_targets("covid_19", region_name)

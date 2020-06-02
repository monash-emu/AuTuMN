"""
Calculates the model outputs for all scenarios for a set of MCMC calibration outputs.
"""
import logging
from autumn.db import Database
from autumn.tool_kit.params import update_params
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.timer import Timer
from autumn.db.models import store_run_models


META_COLS = ["idx", "Scenario", "loglikelihood", "accept"]

logger = logging.getLogger(__file__)


def run_full_models_for_mcmc(
    burn_in: int, src_db_path: str, dest_db_path: str, build_model, params: dict
):
    """
    Run the full baseline model and all scenarios for all accepted MCMC runs in src db.
    """
    src_db = Database(src_db_path)
    dest_db = Database(dest_db_path)

    logger.info("Copying mcmc_run table to %s", dest_db_path)

    mcmc_run_df = src_db.db_query("mcmc_run")

    # Apply burn in and save to destination
    burned_runs_str = ", ".join(mcmc_run_df[:burn_in].idx)
    logger.info("Burned MCMC runs %s", burned_runs_str)
    mcmc_run_df = mcmc_run_df[burn_in:]
    dest_db.dump_df("mcmc_run", mcmc_run_df)

    mcmc_runs = list(mcmc_run_df.T.to_dict().values())
    for mcmc_run in mcmc_runs:
        meta = {k: v for k, v in mcmc_run.items() if k in META_COLS}
        if not meta["accept"]:
            logger.info("Ignoring non-accepted MCMC run %s", meta["idx"])
            continue

        logger.info("Running full model for MCMC run %s", meta["idx"])
        param_updates = {k: v for k, v in mcmc_run.items() if k not in META_COLS}

        run_idx = meta["idx"].split("_")[-1]

        def update_func(ps: dict):
            return update_params(ps, param_updates)

        with Timer("Running model scenarios"):
            num_scenarios = 1 + len(params["scenarios"].keys())
            scenarios = []
            for scenario_idx in range(num_scenarios):
                scenario = Scenario(build_model, scenario_idx, params)
                scenarios.append(scenario)

            # Run the baseline scenario.
            baseline_scenario = scenarios[0]
            baseline_scenario.run(update_func=update_func)
            baseline_model = baseline_scenario.model

            # Run all the other scenarios
            for scenario in scenarios[1:]:
                scenario.run(base_model=baseline_model, update_func=update_func)

        with Timer("Saving model outputs to the database"):
            models = [s.model for s in scenarios]
            store_run_models(models, dest_db_path, run_idx=run_idx)

    logger.info("Finished running full models for all accepted MCMC runs.")

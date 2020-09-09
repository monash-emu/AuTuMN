"""
Calculates the model outputs for all scenarios for a set of MCMC calibration outputs.
"""
import logging
from autumn.db import Database
from autumn.tool_kit.params import update_params
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.timer import Timer
from autumn import db


META_COLS = ["idx", "Scenario", "loglikelihood", "accept"]

logger = logging.getLogger(__name__)


def run_full_models_for_mcmc(
    burn_in: int, src_db_path: str, dest_db_path: str, build_model, params: dict
):
    """
    Run the full baseline model and all scenarios for all accepted MCMC runs in src db.
    """
    src_db = Database(src_db_path)
    dest_db = Database(dest_db_path)
    db.process.apply_burn_in(src_db, dest_db, burn_in)
    mcmc_run_df = dest_db.query("mcmc_run")
    for idx, mcmc_run in mcmc_run_df.iterrows():
        run_id = mcmc_run["run"]
        chain_id = mcmc_run["chain"]
        if not mcmc_run["accept"]:
            logger.info("Ignoring non-accepted MCMC run %s", run_id)
            continue

        logger.info("Running full model for MCMC run %s", run_id)
        param_updates = db.load.load_mcmc_params(dest_db, run_id)
        update_func = lambda ps: update_params(ps, param_updates)
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
            db.store.store_run_models(
                models, dest_db_path, run_id=int(run_id), chain_id=int(chain_id)
            )

    logger.info("Finished running full models for all accepted MCMC runs.")

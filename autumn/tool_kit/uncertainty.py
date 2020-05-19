import pandas as pd
import numpy as np
from typing import List


def collect_iteration_weights(mcmc_tables: List[pd.DataFrame], burn_in=0):
    """
    Work out the weights associated with accepted iterations, considering how many rejections followed each acceptance
    :param mcmc_tables: list of mcmc output tables
    :param burn_in: number of discarded iterations
    :return: list of dictionaries (one dictionary per MCMC chain)
        keys are the run ids and values are the iteration weights
    """
    weights = []
    for i_chain in range(len(mcmc_tables)):
        mcmc_tables[i_chain].sort_values(["idx"])
        weight_dict = {}
        last_run_id = None
        for i_row, run_id in enumerate(mcmc_tables[i_chain].idx):
            if int(run_id[4:]) < burn_in:
                continue
            if mcmc_tables[i_chain].accept[i_row] == 1:
                weight_dict[run_id] = 1
                last_run_id = run_id
            elif last_run_id is None:
                continue
            else:
                weight_dict[last_run_id] += 1
        weights.append(weight_dict)
    return weights


def compute_mcmc_output_quantiles(mcmc_tables: List[pd.DataFrame], output_tables: List[pd.DataFrame],
                                  derived_output_tables: List[pd.DataFrame], weights: List[int], output_name: str,
                                  q_list=[.025, .25, .5, .75, .975]):
    """

    :param mcmc_tables:
    :param output_tables:
    :param derived_output_tables:
    :param weights:
    :param output_name:
    :param q_list:
    :return:
    """
    # find the earliest time that is common to all accepted runs (if start_time was varied)
    if "start_time" in mcmc_tables[0].columns:
        t_min = round(
            max(
                [
                    max(mcmc_tables[i].start_time[mcmc_tables[i].accept == 1])
                    for i in range(len(mcmc_tables))
                ]
            )
        )
    else:
        t_min = output_tables[0].times[0]
    t_max = list(output_tables[0].times)[-1]

    times = list(np.linspace(t_min, t_max, num=t_max - t_min + 1))
    quantiles = np.zeros((len(times), 5))
    for i, time in enumerate(times):
        output_list = []
        for i_chain in range(len(mcmc_tables)):
            for run_id, w in weights[i_chain].items():
                output_list += [
                    float(
                        derived_output_tables[i_chain][output_name][
                            (derived_output_tables[i_chain].idx == run_id)
                            & (derived_output_tables[i_chain].times == time)
                        ]
                    )
                ] * w
        quantiles[i, :] = np.quantile(output_list, q_list)
    return times, quantiles

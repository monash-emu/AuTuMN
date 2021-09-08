"""
Utility functions specific to the ManagedCalibrationRun

Specifically, all functions in here must use only the outputs of ManagedCalibratioRun,
eg the mcmc_runs and mcmc_params DataFrames in the format supplied
"""

import pandas as pd

def get_posteriors(mcmc_params: pd.DataFrame, mcmc_runs: pd.DataFrame, burn:int = 0) -> pd.DataFrame:
    """Get all posteriors for the given run, based on the supplied DataFrames as returned
    by ManagedCalibrationRun

    Args:
        mcmc_params (pd.DataFrame): MCMC Parameters
        mcmc_runs (pd.DataFrame): MCMC Runs
        burn (int, optional): Number of initial runs to burn

    Returns:
        pd.DataFrame: Posteriors
    """
    runs = mcmc_runs[mcmc_runs['run'] >= burn]
    runs = runs[runs['accept']==1]
    
    all_posteriors = []
    
    for idx, run in runs.iterrows():
        all_posteriors += [mcmc_params.loc[idx]] * int(run['weight'])
   
    return pd.DataFrame(all_posteriors)

def get_posterior(mcmc_params: pd.DataFrame, mcmc_runs: pd.DataFrame, param: str, burn:int = 0) -> pd.Series:
    """Get a single posterior (as specified by param) based on the supplied DataFrames as returned
    by ManagedCalibrationRun.  

    Note this exists only as a convenience function; if you require multiple posteriors, it's much faster
    to use get_posteriors (see above)

    Args:
        mcmc_params (pd.DataFrame): MCMC Parameters
        mcmc_runs (pd.DataFrame): MCMC Runs
        param (str): The param (column name) to select
        burn (int, optional): Number of initial runs to burn

    Returns:
        pd.Series: Posterior
    """
    runs = mcmc_runs[mcmc_runs['run'] >= burn]
    runs = runs[runs['accept']==1]
    
    param_df = mcmc_params[param]
    
    all_posteriors = []
    
    for idx, run in runs.iterrows():
        all_posteriors += [param_df.loc[idx]] * int(run['weight'])
   
    return pd.Series(all_posteriors)

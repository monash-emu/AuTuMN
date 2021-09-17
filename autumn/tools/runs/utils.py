"""
Miscellaneous helpers used by ManagedRun and its friends
"""

import pandas as pd

def website_addr_for_run(run_id: str) -> str:
    """
    Return the autumn-data.com URL for a given run_id
    """
    app, region, ts, sha = run_id.split('/')
    return f"http://autumn-data.com/app/{app}/region/{region}/run/{ts}-{sha}.html"

def collate_columns_to_urun(df: pd.DataFrame, drop=False) -> pd.DataFrame:
    """Converts (in place) a concatenated DataFrame with columns ['run', 'chain'] 
    to one with universal unique run ids (urun) of format '{chain}_{run}'
    These are expanded to 2 and 6 characters respectively, just to ensure sort order
    is something that users might expect

    Args:
        df (pd.DataFrame): The DataFrame to convert
        drop (bool, optional): Drop the existing ['run','chain'] columns

    Returns:
        [pd.DataFrame]: The resulting DataFrame
    """
    cols = ['chain', 'run']    # Set columns to combine
    df['urun'] = df['chain'].astype(str).str.zfill(2) + '_' + df['run'].astype(str).str.zfill(6)
    if drop:
        df = df.drop(columns=cols)
    return df

from datetime import date, datetime

import pandas as pd
from autumn.core.inputs.database import get_input_db


def get_kir_vac_coverage() -> pd.Series:
    """
    Returns:
        pd.Series: A Pandas series of year and coverage values
    """

    input_db = get_input_db()

    df = input_db.query("tb_kir_bcg_cov", columns=["year", "prop_coverage"])

    # Calculate the coverage
    vac_dates = df["year"].to_numpy()
    vac_coverage = df["coverage"].to_numpy()

    coverage_too_large = any(vac_coverage >= 0.99)

    # Validation
    if any([coverage_too_large]):
        AssertionError("Unrealistic coverage")

    return pd.Series(vac_coverage, index=vac_dates)

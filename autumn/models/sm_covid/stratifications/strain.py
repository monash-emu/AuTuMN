import pandas as pd
from datetime import date

from autumn.core.inputs.database import get_input_db


def get_first_variant_report_date(variant: str, country: str):
    """
    Determines the first report date of a given variant in a given country

    Args:
        variant: Name of the variant ('delta', 'omicron')
        country: Full name of the country

    Returns:
        Date of first report
    """
    variants_map = {
        "delta": "VOC Delta GK (B.1.617.2+AY.*) first detected in India",
        "omicron": "VOC Omicron GRA (B.1.1.529+BA.*) first detected in Botswana/Hong Kong/South Africa"
    }

    variants_global_emergence_date = {
        "delta": date(2020, 10, 1),   # October 2020 according to WHO 
        "omicron": date(2021, 11, 1)  # November 2021 according to WHO
    }

    assert variant in variants_map, f"Variant {variant} not available from current GISAID database"

    input_db = get_input_db()
    report_dates = input_db.query(
        table_name='gisaid', 
        conditions={"Country": country, "Value": variants_map[variant]},
        columns=["Week prior to"]
    )["Week prior to"]

    if len(report_dates) == 0:
        return None

    first_report_date = report_dates.min()    
    assert first_report_date >= variants_global_emergence_date[variant], "First report precedes global variant emergence"
    
    return first_report_date
    
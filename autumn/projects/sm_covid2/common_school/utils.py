import yaml
import os

from pathlib import Path

import pandas as pd

from autumn.settings.folders import PROJECTS_PATH, INPUT_DATA_PATH

SCHOOL_ISO2_LIST = [
    'AW', 'AF', 'AO', 'AL', 'AE', 'AR', 'AM', 'AG', 'AU', 'AT', 'AZ', 'BI', 'BE', 'BJ', 'BF', 'BD', 'BG', 'BH', 'BS', 
    'BA', 'BY', 'BZ', 'BO', 'BR', 'BB', 'BN', 'BW', 'CF', 'CA', 'CH', 'CL', 'CN', 'CI', 'CM', 'CD', 'CG', 'CO', 'KM', 
    'CV', 'CR', 'CU', 'CW', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DO', 'DZ', 'EC', 'EG', 'ES', 'EE', 'ET', 'FI', 'FJ', 'FR', 
    'GA', 'GB', 'GE', 'GH', 'GN', 'GM', 'GQ', 'GR', 'GD', 'GT', 'GY', 'HN', 'HR', 'HT', 'HU', 'ID', 'IN', 'IE', 'IR', 
    'IQ', 'IS', 'IL', 'IT', 'JM', 'JO', 'JP', 'KZ', 'KE', 'KG', 'KH', 'KI', 'KR', 'KW', 'LA', 'LB', 'LR', 'LY', 'LC', 
    'LK', 'LS', 'LT', 'LU', 'LV', 'MA', 'MD', 'MV', 'MX', 'MK', 'ML', 'MT', 'MM', 'ME', 'MN', 'MZ', 'MR', 'MU', 'MW', 
    'MY', 'NA', 'NE', 'NG', 'NI', 'NL', 'NO', 'NP', 'NZ', 'OM', 'PK', 'PA', 'PE', 'PH', 'PG', 'PL', 'PT', 'PY', 'PS', 
    'QA', 'RO', 'RU', 'RW', 'SA', 'SD', 'SN', 'SG', 'SB', 'SL', 'SV', 'RS', 'SS', 'ST', 'SR', 'SK', 'SI', 'SE', 'SZ', 
    'SC', 'SY', 'TD', 'TG', 'TH', 'TL', 'TT', 'TN', 'TR', 'TZ', 'UG', 'UA', 'UY', 'US', 'UZ', 'VC', 'VE', 'VN', 'ZA', 
    'ZM', 'ZW'
]

def get_owid_data(columns, iso_code = None):
    in_file = Path(INPUT_DATA_PATH) / "owid.parquet"

    df = pd.read_parquet(in_file, columns=columns)

    if iso_code is not None:
        df = df[df["iso_code"] == iso_code]
        df.drop(columns=["iso_code"], inplace=True)
    
    return df
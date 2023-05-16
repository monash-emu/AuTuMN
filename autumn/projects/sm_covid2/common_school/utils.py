import yaml
import os
from autumn.settings.folders import PROJECTS_PATH


def get_school_country_list(nationbal_only=False):
    source = os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")
    country_dict = yaml.load(open(source), Loader=yaml.UnsafeLoader)
    primary_key = "national" if nationbal_only else "all"

    return country_dict[primary_key]   


SCHOOL_ISO2_LIST = [
    'AL', 'AE', 'AR', 'AU', 'AT', 'BE', 'BJ', 'BF', 'BD', 'BA', 'BR', 'CA', 'CH', 'CL', 'CN', 
    'CM', 'CD', 'CG', 'CO', 'CZ', 'DE', 'DK', 'DO', 'EC', 'EG', 'ES', 'EE', 'ET', 'FI', 'FR',
    'GA', 'GB', 'GH', 'GR', 'HN', 'HR', 'HU', 'ID', 'IN', 'IE', 'IR', 'IQ', 'IS', 'IL', 'IT', 
    'JO', 'JP', 'KZ', 'KE', 'KR', 'LB', 'LT', 'LU', 'MV', 'MX', 'MK', 'ML', 'MN', 'MZ', 'MW', 
    'MY', 'NG', 'NL', 'NO', 'NP', 'OM', 'PK', 'PE', 'PH', 'PL', 'PT', 'PY', 'PS', 'QA', 'RO', 
    'RU', 'SA', 'SN', 'SG', 'SL', 'SS', 'SI', 'SE', 'US', 'UZ', 'ZA', 'ZM', 'ZW'
]
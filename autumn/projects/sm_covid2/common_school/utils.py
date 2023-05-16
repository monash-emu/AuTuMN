import yaml
import os

import country_converter as coco

from autumn.settings.folders import PROJECTS_PATH


def get_school_country_list(nationbal_only=False):
    source = os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")
    country_dict = yaml.load(open(source), Loader=yaml.UnsafeLoader)
    primary_key = "national" if nationbal_only else "all"

    return country_dict[primary_key]   


def get_school_iso2_list(national_only=False):
    country_dict = get_school_country_list(national_only)
    return coco.convert(names=list(country_dict.keys()), to='ISO2') 

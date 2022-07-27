import pandas as pd
from datetime import date
from summer import CompartmentalModel
from typing import Dict

from autumn.core.inputs.database import get_input_db
from autumn.models.sm_covid.parameters import VocComponent
from autumn.settings.constants import COVID_BASE_DATETIME

# Improt modules from the sm_sir model
from autumn.models.sm_sir.stratifications.strains import make_voc_seed_func


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
    

def seed_vocs_using_gisaid(model: CompartmentalModel, all_voc_params: Dict[str, VocComponent], seed_compartment: str, country_name: str):
    """
    Use importation flows to seed VoC cases.

    Generally seeding to the infectious compartment, because unlike Covid model, this compartment always present.

    Note that the entry rate will get repeated for each compartment as the requested compartments for entry are
    progressively stratified after this process is applied (but are split over the previous stratifications of the
    compartment to which this is applied, because the split_imports argument is True).

    Args:
        model: The summer model object
        all_voc_params: The VoC-related parameters
        seed_compartment: The compartment that VoCs should be seeded to
        country_name: The modelled country's name
    """

    for voc_name, this_voc_params in all_voc_params.items():
        voc_seed_params = this_voc_params.new_voc_seed
        if voc_seed_params:
            # work out seed time using gisaid data
            first_report_date = get_first_variant_report_date(voc_name, country_name)
            first_report_date_as_int = (first_report_date - COVID_BASE_DATETIME).days
            seed_time = first_report_date_as_int + voc_seed_params.time_from_gisaid_report

            voc_seed_func = make_voc_seed_func(
                voc_seed_params.entry_rate,
                seed_time,
                voc_seed_params.seed_duration
            )
            model.add_importation_flow(
                f"seed_voc_{voc_name}",
                voc_seed_func,
                dest=seed_compartment,
                dest_strata={"strain": voc_name},
                split_imports=True
            )
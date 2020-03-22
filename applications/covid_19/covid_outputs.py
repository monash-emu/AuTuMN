from autumn.constants import Compartment
from datetime import date


def find_incidence_outputs(parameters):
    last_presympt = \
        'presympt_' + str(parameters['n_compartment_repeats']) if \
            parameters['n_compartment_repeats'] > 1 else \
            'presympt'
    first_infectious = \
        'infectious_1' if \
            parameters['n_compartment_repeats'] > 1 else \
            'infectious'
    return {
        'incidence':
            {'origin': last_presympt,
             'to': first_infectious,
             'origin_condition': '',
             'to_condition': ''
             }
    }


def create_request_stratified_incidence_covid(requested_stratifications, strata_dict, n_compartment_repeats):
    """
    Create derived outputs for disaggregated incidence
    """
    out_connections = {}
    origin_compartment = \
        Compartment.PRESYMPTOMATIC if \
            n_compartment_repeats < 2 else \
            Compartment.PRESYMPTOMATIC + '_' + str(n_compartment_repeats)
    to_compartment = \
        Compartment.INFECTIOUS if \
            n_compartment_repeats < 2 else \
            Compartment.INFECTIOUS + '_1'
    for stratification in requested_stratifications:
        for stratum in strata_dict[stratification]:
            out_connections['incidenceX' + stratification + '_' + stratum] \
                = {
                'origin': origin_compartment,
                'to': to_compartment,
                'origin_condition': '',
                'to_condition': stratification + '_' + stratum,
            }
    return out_connections


def find_date_from_year_start(times, incidence):
    """
    Messy patch to shift dates over such that zero represents the start of the year and the number of cases are
    approximately correct for Australia at 22nd March
    """
    year, month, day = 2020, 3, 22
    cases = 1098.
    data_days_from_year_start = (date(year, month, day) - date(year, 1, 1)).days
    model_days_reach_target = next(i_inc[0] for i_inc in enumerate(incidence) if i_inc[1] > cases)
    days_to_add = data_days_from_year_start - model_days_reach_target
    return [int(i_time) + days_to_add for i_time in times]

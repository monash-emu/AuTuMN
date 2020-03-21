from autumn.constants import Compartment


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

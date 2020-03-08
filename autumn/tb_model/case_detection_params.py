from autumn.tb_model import convert_competing_proportion_to_rate
from autumn.tool_kit import return_function_of_function


def find_organ_specific_cdr(cdr_scaleup_raw, parameters, organ_strata, target_organ_props):
    
    # Disease duration by organ
    overall_duration = target_organ_props['smearpos'] * 1.6 + 5.3 * (1 - target_organ_props['smearpos'])
    disease_duration = {
        "smearpos": 1.6,
        "smearneg": 5.3,
        "extrapul": 5.3,
        "overall": overall_duration,
    }

    # work out the CDR for smear-positive TB
    def cdr_smearpos(time):
        return cdr_scaleup_raw(time) / (
                target_organ_props['smearpos']
                + target_organ_props['smearneg'] * parameters["diagnostic_sensitivity_smearneg"]
                + target_organ_props['extrapul'] * parameters["diagnostic_sensitivity_extrapul"]
        )

    def cdr_smearneg(time):
        return cdr_smearpos(time) * parameters["diagnostic_sensitivity_smearneg"]

    def cdr_extrapul(time):
        return cdr_smearpos(time) * parameters["diagnostic_sensitivity_extrapul"]

    cdr_by_organ = {
        "smearpos": cdr_smearpos,
        "smearneg": cdr_smearneg,
        "extrapul": cdr_extrapul,
        "overall": cdr_scaleup_raw,
    }
    detect_rate_by_organ = {}
    for organ in organ_strata + ['overall']:
        prop_to_rate = convert_competing_proportion_to_rate(1. / disease_duration[organ])
        detect_rate_by_organ[organ] = return_function_of_function(cdr_by_organ[organ], prop_to_rate)

    return detect_rate_by_organ

from summer2 import Stratification, Multiply
from autumn.models.tb_dynamics2.constants import BASE_COMPARTMENTS
from autumn.models.tb_dynamics2.parameters import Parameters


def get_gender_strat(
        params: Parameters
    ) -> Stratification:
    """
    Stratify all model compartments based on a user-defined stratification request.
    """
    requested_strata = params.gender.strata
    props = params.gender.proportions
    strat = Stratification("gender", requested_strata, BASE_COMPARTMENTS)
    strat.set_population_split(props)
    # Pre-process generic flow adjustments:
    # IF infection is adjusted and other infection flows NOT adjusted
    # THEN use the infection adjustment for the other infection flows

    adjustments = vars(params.gender.adjustments)
    adjs = {}
    if 'infection' in adjustments.keys():
        inf_adjs = vars(params.gender.adjustments.infection)
        item = {'infection': {k: v.value for k,v in inf_adjs.items()}}
        adjs.update(item)
        for stage in ["latent", "recovered"]:
            flow_name = f"infection_from_{stage}"
            if flow_name not in adjs:
                adjs[flow_name] = adjs['infection']
   
    # # # Set generic flow adjustments
    # # # Do not adjust for age under 15    
    for age in params.age_breakpoints:
        for flow_name, adjustment in adjs.items():
            if flow_name != 'birth':
                if age < 15:
                    adj = {k: Multiply(1.0) for k in adjustment.keys()}
                else:
                    adj = {k: Multiply(v) for k, v in adjustment.items()}
                strat.set_flow_adjustments(flow_name, adj, source_strata={"age": str(age)})
    return strat
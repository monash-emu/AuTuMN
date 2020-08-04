from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from summer.model.utils.flowchart import create_flowchart

from autumn.curve import scale_up_function


def build_model(params: dict, update_params={}) -> StratifiedModel:
    """
    Build the master function to run a simple SIR model

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """
    params.update(update_params)
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED
    ]

    params['delta'] = params['beta'] * params['rr_reinfection_once_recovered']

    params['theta'] = params['beta'] * params['rr_reinfection_once_infected']  # not used at the moment

    flows = [
        {
            'type': 'infection_frequency',
            'parameter': 'beta',
            'origin': Compartment.SUSCEPTIBLE,
            'to': Compartment.EARLY_LATENT
        },
        {
            'type': 'standard_flows',
            'parameter': 'kappa',
            'origin': Compartment.EARLY_LATENT,
            'to': Compartment.LATE_LATENT
        },
        {
            'type': 'standard_flows',
            'parameter': 'epsilon',
            'origin': Compartment.EARLY_LATENT,
            'to': Compartment.INFECTIOUS
        },
        
        {
            'type': 'standard_flows',
            'parameter': 'nu',
            'origin': Compartment.LATE_LATENT,
            'to': Compartment.INFECTIOUS
        },
        {
            'type': 'standard_flows',
            'parameter': 'tau',
            'origin': Compartment.INFECTIOUS,
            'to': Compartment.RECOVERED
        },
        {
            'type': 'standard_flows',
            'parameter': 'gamma',
            'origin': Compartment.INFECTIOUS,
            'to': Compartment.LATE_LATENT
        },
        {
            'type': 'standard_flows',
            'parameter': 'delta',
            'origin': Compartment.RECOVERED,
            'to': Compartment.EARLY_LATENT
        },
        {
            'type': 'compartment_death',
            'parameter': 'infect_death',
            'origin': Compartment.INFECTIOUS
        }

    ]

    integration_times = get_model_times_from_inputs(
        round(params["start_time"]), params["end_time"], params["time_step"]
    )

    init_conditions = {Compartment.INFECTIOUS: 1}

    tb_sir_model = StratifiedModel(
        integration_times,
        compartments,
        init_conditions,
        params,
        requested_flows=flows,
        birth_approach=BirthApproach.REPLACE_DEATHS, 
        entry_compartment='susceptible', 
        starting_population=100000,
        infectious_compartment=(Compartment.INFECTIOUS,), 
        verbose=True,
        
    )

    tb_sir_model.adaptation_functions['universal_death_rateX'] = lambda x: 1./70

# #   add  time_variant parameters

    def time_variant_CDR():
        return scale_up_function(params['cdr'].keys(), params['cdr'].values(), smoothness=0.2, method=5)

    def time_variant_TSR():
        return scale_up_function(params['tsr'].keys(), params['tsr'].values(), smoothness=0.2, method=5)

    def time_variant_DST():
        return scale_up_function(params['dst'].keys(), params['dst'].values(), smoothness=0.2, method=5)

    my_tv_CDR = time_variant_CDR()
    my_tv_TSR = time_variant_TSR()
    my_tv_DST = time_variant_DST()

    def my_tv_tau(time):
        return my_tv_detection_rate(time) * my_tv_TSR(time) * my_tv_DST(time)

    tb_sir_model.adaptation_functions["tau"] = my_tv_tau
    tb_sir_model.parameters["tau"] = "tau"

    def my_tv_detection_rate(time):
        return my_tv_CDR(time)/(1-my_tv_CDR(time)) * (params['gamma'] + params['universal_death_rate'] + params['infect_death']) #calculating the time varaint detection rate from tv CDR


   #adding strain stratification

    stratify_by = ['strain']  # ['strain', 'treatment_type']

    if 'strain' in stratify_by:

        tb_sir_model.stratify(
            "strain", ['ds', 'inh_R', 'rif_R', 'mdr'],
            compartment_types_to_stratify=[Compartment.EARLY_LATENT, Compartment.LATE_LATENT, Compartment.INFECTIOUS],
            requested_proportions={'ds': 1., 'inh_R': 0., 'rif_R':0., 'mdr': 0.}, #adjustment_requests={'tau': tau_adjustment},
            verbose=False,
            adjustment_requests={'beta': {'ds': 1., 'inh_R': 0.9, 'rif_R': 0.7, 'mdr': 0.5}})


        # set up some amplification flows
        # INH_R
        tb_sir_model.add_transition_flow(
                    {"type": "standard_flows", "parameter": "dr_amplification_inh",
                     "origin": "infectiousXstrain_ds", "to": "infectiousXstrain_inh_R",
                     "implement": len(tb_sir_model.all_stratifications)})

        def tv_amplification_inh_rate(time):
            return my_tv_detection_rate(time) * (1 - my_tv_TSR(time)) * params['prop_of_failures_developing_inh_R']

        tb_sir_model.adaptation_functions["dr_amplification_inh"] = tv_amplification_inh_rate
        tb_sir_model.parameters["dr_amplification_inh"] = "dr_amplification_inh"


        # set up amplification flow for RIF_R
        tb_sir_model.add_transition_flow(
                    {"type": "standard_flows", "parameter": "dr_amplification_rif",
                     "origin": "infectiousXstrain_ds", "to": "infectiousXstrain_rif_R",
                     "implement": len(tb_sir_model.all_stratifications)})

        def tv_amplification_rif_rate(time):
            return my_tv_detection_rate(time) * (1 - my_tv_TSR(time)) * params['prop_of_failures_developing_rif_R']

        tb_sir_model.adaptation_functions["dr_amplification_rif"] = tv_amplification_rif_rate
        tb_sir_model.parameters["dr_amplification_rif"] = "dr_amplification_rif"


        # set up amplification flow for MDR

        tb_sir_model.add_transition_flow(
                {"type": "standard_flows", "parameter": "dr_amplification_rif",
                 "origin": "infectiousXstrain_inh_R", "to": "infectiousXstrain_mdr",
                 "implement": len(tb_sir_model.all_stratifications)})

    
        tb_sir_model.add_transition_flow(
                {"type": "standard_flows", "parameter": "dr_amplification_inh",
                 "origin": "infectiousXstrain_rif_R", "to": "infectiousXstrain_mdr",
                 "implement": len(tb_sir_model.all_stratifications)})


    # tb_sir_model.transition_flows.to_csv("transitions.csv")

    # create_flowchart(tb_sir_model, name="sir_model_diagram")

    tb_sir_model.run_model()

    return tb_sir_model

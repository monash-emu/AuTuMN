from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach


def build_model(country: str, params: dict, update_params={}):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """
    params.update(update_params)

    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    ]

    flows = [
        {
            'type': 'infection_frequency',
            'parameter': 'contact_rate',
            'origin': Compartment.SUSCEPTIBLE,
            'to': Compartment.INFECTIOUS
        },
        {
            'type': 'standard_flows',
            'parameter': 'recovery_rate',
            'origin': Compartment.INFECTIOUS,
            'to': Compartment.RECOVERED
        },
    ]

    integration_times = list(range(100))
    init_conditions = {Compartment.INFECTIOUS: 1}

    sir_model = StratifiedModel(
        integration_times,
        compartments,
        init_conditions,
        params,
        flows,
        infectious_compartment=(Compartment.INFECTIOUS,),
        birth_approach=BirthApproach.NO_BIRTH,
        starting_population=1000000,
    )

    return sir_model

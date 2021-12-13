from typing import Optional, Dict

from summer import StrainStratification, Multiply

from autumn.models.covid_19.parameters import VocComponent
from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, Strain, INFECTION


def get_strain_strat(voc_params: Optional[Dict[str, VocComponent]]):
    """
    Stratify the model by strain, with two strata, being wild or "ancestral" virus type and the single variant of
    concern ("VoC").

    Args:
        voc_params: All the VoC parameters (one VocComponent parameters object for each VoC)

    Returns:
        The strain stratification summer object

    """

    # Process the requests
    voc_names = list(voc_params.keys())

    # Stratify model
    strain_strat = StrainStratification("strain", [Strain.WILD_TYPE] + voc_names, DISEASE_COMPARTMENTS)

    # Prepare population split and transmission adjustments
    population_split = {Strain.WILD_TYPE: 1.}
    transmissibility_adjustment = {Strain.WILD_TYPE: None}
    for voc_name in voc_names:
        population_split[voc_name] = 0.
        transmissibility_adjustment[voc_name] = Multiply(voc_params[voc_name].contact_rate_multiplier)

    # Apply population split
    strain_strat.set_population_split(population_split)

    # Apply transmissibility adjustments
    strain_strat.set_flow_adjustments(INFECTION, transmissibility_adjustment)

    return strain_strat

from summer import StrainStratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, Strain, INFECTION


def get_strain_strat(voc_params):
    """
    Stratify the model by strain, with two strata, being wild or "ancestral" virus type and the single
    variant of concern ("VoC").
    """

    # Process the requests
    voc_names = list(voc_params.keys())
    msg = f"VoC names must be different from: {Strain.WILD_TYPE}"
    assert Strain.WILD_TYPE not in voc_names, msg

    # Stratify model
    strain_strat = StrainStratification("strain", [Strain.WILD_TYPE] + voc_names, DISEASE_COMPARTMENTS)

    # Prepare population split and transmission adjustments.
    population_split = {Strain.WILD_TYPE: 1.}
    transmissibility_adjustment = {Strain.WILD_TYPE: None}
    for voc_name in voc_names:
        population_split[voc_name] = 0.
        transmissibility_adjustment[voc_name] = Multiply(voc_params[voc_name].contact_rate_multiplier)

    # Apply population split
    strain_strat.set_population_split(population_split)

    # Apply transmissibility adjustments
    strain_strat.add_flow_adjustments(INFECTION, transmissibility_adjustment)

    return strain_strat

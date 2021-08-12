from summer import StrainStratification, Multiply

from autumn.models.covid_19.constants import DISEASE_COMPARTMENTS, Strain


def get_strain_strat(voc_params):
    """
    Stratify the model by strain, with two strata, being wild or "ancestral" virus type and the single
    variant of concern ("VoC").
    """
    voc_names = list(voc_params.keys())

    msg = f"VoC names must be different from {Strain.WILD_TYPE}"
    assert Strain.WILD_TYPE not in voc_names, msg

    # Stratify model.
    strain_strat = StrainStratification(
        "strain",
        [Strain.WILD_TYPE] + voc_names,
        DISEASE_COMPARTMENTS,
    )

    # Prepare population split and transmission adjustments.
    population_split = {Strain.WILD_TYPE: 1.}
    trans_adjustment = {Strain.WILD_TYPE: None}
    for voc_name in voc_names:
        population_split[voc_name] = 0.
        trans_adjustment[voc_name] = Multiply(voc_params[voc_name].contact_rate_multiplier)

    # Apply population split
    strain_strat.set_population_split(population_split)

    # Apply transmission adjustments
    strain_strat.add_flow_adjustments("infection", trans_adjustment)

    return strain_strat


def make_voc_seed_func(entry_rate, start_time, seed_duration):
    def voc_seed_func(time, computed_values):
        return entry_rate if 0. < time - start_time < seed_duration else 0.

    return voc_seed_func
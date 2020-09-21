from autumn.tool_kit import schema_builder as sb

validate_params = sb.build_validator(
    # Country info
    iso3=str,
    region=sb.Nullable(str),
    # Running time.
    start_time=float,
    end_time=float,
    time_step=float,
    # output requests
    calculated_outputs=list,
    outputs_stratification=dict,
    # model structure,
    stratify_by=list,
    age_breakpoints=list,
    user_defined_stratifications=dict,
    # demographics
    start_population_size=float,
    universal_death_rate=float,
    # base disease model
    contact_rate=float,
    age_specific_latency=dict,
    self_recovery_rate_dict=dict,
    infect_death_rate_dict=dict,
    rr_infection_latent=float,
    rr_infection_recovered=float,
    # detection
    time_variant_tb_screening_rate=dict,
    passive_screening_sensitivity=dict,
    # treatment
    treatment_duration=float,
    time_variant_tsr=dict,
    prop_death_among_negative_tx_outcome=float,
    on_treatment_infect_multiplier=float,
    # characterising age stratification
    age_infectiousness_switch=float,
    # defining organ stratification
    incidence_props_pulmonary=float,
    incidence_props_smear_positive_among_pulmonary=float,
    smear_negative_infect_multiplier=float,
    extrapulmonary_infect_multiplier=float,
    # other
    inflate_reactivation_for_diabetes=bool,
    extra_params=dict,
)


def check_param_values(params):
    assert all([v > 0. for v in params['time_variant_tsr'].values()]), "Treatment success rate should always be > 0."

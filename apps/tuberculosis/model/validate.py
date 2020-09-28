from autumn.tool_kit import schema_builder as sb

validate_params = sb.build_validator(
    # Country info
    iso3=str,
    region=sb.Nullable(str),
    # Running time.
    start_time=float,
    end_time=float,
    time_step=float,
    critical_ranges=list,
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
    late_reactivation_multiplier=float,
    self_recovery_rate_dict=dict,
    infect_death_rate_dict=dict,
    rr_infection_latent=float,
    rr_infection_recovered=float,
    time_variant_bcg_perc=dict,
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
    # interventions
    time_variant_acf=list,
    acf_screening_sensitivity=float,
    time_variant_ltbi_screening=list,
    ltbi_screening_sensitivity=float,
    pt_efficacy=float,
    # other
    inflate_reactivation_for_diabetes=bool,
    extra_params=dict,
    # dispersion parameters for MCMC calibration  # FIXME: we should avoid this
    prevalence_infectiousXlocation_majuro_dispersion_param=float,
    prevalence_infectiousXlocation_ebeye_dispersion_param=float,
    percentage_latentXlocation_majuro_dispersion_param=float,
    notificationsXlocation_majuro_dispersion_param=float,
    notificationsXlocation_ebeye_dispersion_param=float,
    notificationsXlocation_other_dispersion_param=float,
    population_size_dispersion_param=float,
)


def check_param_values(params):
    assert all([v > 0. for v in params['time_variant_tsr'].values()]), "Treatment success rate should always be > 0."

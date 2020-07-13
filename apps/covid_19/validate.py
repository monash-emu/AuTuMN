from autumn.tool_kit import schema_builder as sb

validate_params = sb.build_validator(
    stratify_by=sb.List(str),
    # Country info
    iso3=str,
    region=sb.Nullable(str),
    # Running time.
    start_time=float,
    end_time=float,
    time_step=float,
    # Compartment construction
    compartment_periods=sb.DictGeneric(str, float),
    compartment_periods_calculated=dict,
    # Infectiousness adjustments (not sure where used)
    hospital_props=sb.List(float),
    hospital_props_multiplier=float,
    # mortality parameters
    use_verity_mortality_estimates=bool,
    infection_fatality_props=sb.List(float),
    ifr_double_exp_model_params=dict,
    # Age stratified params
    agegroup_breaks=sb.List(float),
    age_based_susceptibility=sb.DictGeneric(str, float),
    # Clinical status stratified params
    clinical_strata=sb.List(str),
    non_sympt_infect_multiplier=float,
    late_infect_multiplier=sb.Dict(sympt_isolate=float, hospital_non_icu=float, icu=float),
    icu_mortality_prop=float,
    symptomatic_props=sb.List(float),
    icu_prop=float,
    prop_detected_among_symptomatic=float,
    # Time-variant detection of COVID cases, used to construct a tanh function.
    tv_detection_b=float,
    tv_detection_c=float,
    tv_detection_sigma=float,
    int_detection_gap_reduction=float,
    # Dynamic mixing matrix updates
    # Time-varying mixing matrix adjutment by location
    mixing=sb.DictGeneric(
        str,
        sb.Dict(
            # Whether to append or overwrite times / values
            append=bool,
            # Times for dynamic mixing func.
            times=list,
            # Values for dynamic mixing func.
            values=list,
        ),
    ),
    # Time-varying mixing matrix adjustment by age
    mixing_age_adjust=sb.DictGeneric(
        str,
        sb.Dict(
            # Times for dynamic mixing func.
            times=list,
            # Values for dynamic mixing func.
            values=sb.List(float),
        ),
    ),
    npi_effectiveness=sb.DictGeneric(str, float),
    is_periodic_intervention=bool,
    periodic_intervention=sb.Dict(
        restart_time=float,
        prop_participating=float,
        contact_multiplier=float,
        duration=float,
        period=float,
    ),
    google_mobility_locations=sb.DictGeneric(str, sb.List(str)),
    # Something to do with travellers?
    traveller_quarantine=sb.Dict(times=sb.List(float), values=sb.List(float),),
    # Importation of disease from outside of region
    implement_importation=bool,
    import_secondary_rate=float,
    symptomatic_props_imported=float,
    hospital_props_imported=float,
    icu_prop_imported=float,
    prop_detected_among_symptomatic_imported=float,
    enforced_isolation_effect=float,
    self_isolation_effect=float,
    data=sb.Dict(times_imported_cases=sb.List(float), n_imported_cases=sb.List(float),),
    microdistancing=sb.Nullable(sb.Dict(b=float, c=float, sigma=float)),
    # Other stuff
    contact_rate=float,
    infect_death=float,
    infectious_seed=int,
    universal_death_rate=float,
    # for MCMC calibration with negative binomial likelihood
    notifications_dispersion_param=float,
    prevXlateXclinical_icuXamong_dispersion_param=float,
    infection_deathsXall_dispersion_param=float,
)

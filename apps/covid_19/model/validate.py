from autumn.tool_kit import schema_builder as sb

validate_params = sb.build_validator(
    # Country info
    iso3=str,
    region=sb.Nullable(str),
    pop_region_override=sb.Nullable(str),
    pop_year=int,
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
    use_raw_mortality_estimates=bool,
    infection_fatality_props=sb.List(float),
    ifr_multiplier=float,
    # Age stratified params
    agegroup_breaks=sb.List(float),
    age_based_susceptibility=sb.DictGeneric(str, float),
    # Clinical status stratified params
    late_exposed_infect_multiplier=float,
    non_sympt_infect_multiplier=float,
    late_infect_multiplier=sb.Dict(sympt_isolate=float, hospital_non_icu=float, icu=float),
    icu_mortality_prop=float,
    symptomatic_props=sb.List(float),
    symptomatic_props_multiplier=float,
    icu_prop=float,
    # Time-variant detection of COVID cases, used to construct a tanh function.
    time_variant_detection=sb.Dict(
        maximum_gradient=float, max_change_time=float, start_value=float, end_value=float
    ),
    int_detection_gap_reduction=float,
    # Dynamic mixing matrix updates
    # Time-varying mixing matrix adjustment by location
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
    google_mobility_locations=sb.DictGeneric(str, sb.List(str)),
    smooth_google_data=bool,
    # Something to do with travellers?
    traveller_quarantine=sb.Dict(times=sb.List(float), values=sb.List(float),),
    # Importation of disease from outside of region
    implement_importation=bool,
    import_secondary_rate=float,
    data=sb.Dict(times_imported_cases=sb.List(float), n_imported_cases=sb.List(float),),
    microdistancing=sb.Nullable(sb.Dict(function_type=str, parameters=dict)),
    # Other stuff
    contact_rate=float,
    seasonal_force=sb.Nullable(float),
    infect_death=float,
    infectious_seed=float,
    universal_death_rate=float,
    # for MCMC calibration with negative binomial likelihood
    notifications_dispersion_param=float,
    prevXlate_activeXclinical_icuXamong_dispersion_param=float,
    infection_deathsXall_dispersion_param=float,
    proportion_seropositive_dispersion_param=float,
    hospital_occupancy_dispersion_param=float,
    new_hospital_admissions_dispersion_param=float,
    new_icu_admissions_dispersion_param=float,
    total_infection_deaths_dispersion_param=float,
    # for immunity wane
    full_immunity=bool,
    immunity_duration=float,
    icu_occupancy_dispersion_param=float,
    importation_props_by_age=dict,
    testing_to_detection=sb.Nullable(sb.Dict(maximum_detection=float, shape_parameter=float)),
)

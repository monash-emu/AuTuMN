PARAMETER_DEFINITION = {
    "age_stratification.cfr.source_immunity_distribution.high": 
        "proportion of population received booster in evidence source setting",
    "age_stratification.cfr.source_immunity_distribution.low":
        "proportion of population received two doses but no booster",
    "age_stratification.cfr.source_immunity_distribution.none":
        "proportion of population not received primary course",
    "age_stratification.cfr.source_immunity_protection.high":
        "protection against severe disease from booster assumed for evidence source setting",
    "age_stratification.cfr.source_immunity_protection.low":
        "protection against severe disease from primary course for evidence source setting",
    "asympt_infectiousness_effect":
        "relative infectiousness of asymptomatic persons",
    "booster_effect_duration":
        "duration of booster effect",
    "contact_rate":
        "probability of transmission per contact",
    "hospital_stay.hospital_all.distribution":
        "distribution type for hospital stay",
    "hospital_stay.hospital_all.parameters.mean":
        "mean hospital stay",
    "hospital_stay.hospital_all.parameters.shape":
        "shape parameter for hospital stay",
    "hospital_stay.icu.distribution":
        "distribution type for ICU stay",
    "hospital_stay.icu.parameters.mean":
        "mean ICU stay",
    "hospital_stay.icu.parameters.shape":
        "shape parameter for ICU stay",
    "immunity_stratification.infection_risk_reduction.high":
        "reduction in transmission risk for boosted",
    "immunity_stratification.infection_risk_reduction.low":
        "reduction in transmission risk for primary course",
    "infectious_seed":
        "starting infectious seed",
    "isolate_infectiousness_effect":
        "relative infectiousness of isolated cases",
    "prop_icu_among_hospitalised":
        "proportion of hospitalised persons admitted to ICU",
    "ref_mixing_iso3":
        "ISO3 code for source country for mixing matrix",
    "sojourns.active.proportion_early":
        "proportion of active period before isolation",
    "sojourns.latent.proportion_early":
        "proportion of latent period in first compartment",
    "testing_to_detection.assumed_tests_parameter":
        "index testing rate (\(tests(t)\))",
    "testing_to_detection.assumed_cdr_parameter":
        "CDR reached at index testing rate (\(CDR(t)\))",
    "testing_to_detection.floor_value":
        "CDR floor (\(floor\))",
    "testing_to_detection.smoothing_period":
        "interval for moving average of daily tests",
    "time_from_onset_to_event.death.distribution":
        "disease onset to death distribution",
    "time_from_onset_to_event.death.parameters.mean":
        "mean disease onset to death",
    "time_from_onset_to_event.death.parameters.shape":
        "onset to death shape parameter",
    "time_from_onset_to_event.hospitalisation.distribution":
        "disease onset to hospital admission distribution",
    "time_from_onset_to_event.hospitalisation.parameters.mean":
        "mean disease onset to hospital admission",
    "time_from_onset_to_event.hospitalisation.parameters.shape":
        "onset to hospitalisation shape parameter",
    "time_from_onset_to_event.icu_admission.distribution":
        "disease onset to ICU admission distribution",
    "time_from_onset_to_event.icu_admission.parameters.mean":
        "mean disease onset to ICU admission",
    "time_from_onset_to_event.icu_admission.parameters.shape":
        "onset to ICU shape parameter",
    "time_from_onset_to_event.notification.distribution":
        "disease onset to notification distribution",
    "time_from_onset_to_event.notification.parameters.mean":
        "mean disease onset to notification",
    "time_from_onset_to_event.notification.parameters.shape":
        "onset to notification parameter",
    "sojourns.latent.total_time":
        "infection latent period",
    "sojourns.active.total_time":
        "period with active disease",
    "mobility.microdistancing.behaviour.parameters.max_effect":
        "maximum effect of microdistancing",
}

PARAMETER_EVIDENCE = {
       "hospital_stay.hospital_all.parameters.mean":
    "Mean Hospital stay (ward to discharge) when Omicron dominant variant ranged from 2.05 to 6.02 days across all age groups\\ref{tobin-2022}",
    "hospital_stay.icu.parameters.mean":
    "Mean ICU stay when Omicron was dominant variant ranged from 3.93 to 4.36 days across all age groups\\ref{tobin-2022}",
}

PARAMETER_UNITS = {
    "booster_effect_duration":
        "days",
    "hospital_stay.hospital_all.parameters.mean":
        "days",
    "hospital_stay.icu.parameters.mean":
        "days",
    "infectious_seed":
        "persons",
    "testing_to_detection.assumed_tests_parameter":
        "tests per day",
    "testing_to_detection.smoothing_period":
        "days",
    "time_from_onset_to_event.death.parameters.mean":
        "days",
    "time_from_onset_to_event.hospitalisation.parameters.mean":
        "days",
    "time_from_onset_to_event.icu_admission.parameters.mean":
        "days",
    "sojourns.latent.total_time":
        "days",
    "sojourns.active.total_time":
        "days",
}

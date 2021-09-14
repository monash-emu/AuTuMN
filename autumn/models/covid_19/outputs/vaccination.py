from summer import CompartmentalModel

from autumn.models.covid_19.constants import COMPARTMENTS, Vaccination, PROGRESS, Clinical
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.stratifications.vaccination import VACCINATION_STRATA
from autumn.models.covid_19.outputs.standard import request_stratified_output_for_flow


def request_vaccination_outputs(model: CompartmentalModel, params: Parameters):
    """
    Get the vaccination-related outputs
    """

    # Track proportions vaccinated by vaccination status
    for vacc_stratum in VACCINATION_STRATA:
        model.request_output_for_compartments(
            name=f"_{vacc_stratum}",
            compartments=COMPARTMENTS,
            strata={"vaccination": vacc_stratum},
            save_results=False,
        )
        model.request_function_output(
            name=f"proportion_{vacc_stratum}",
            sources=[f"_{vacc_stratum}", "_total_population"],
            func=lambda vaccinated, total: vaccinated / total,
        )
    model.request_function_output(
        name="at_least_one_dose_prop",
        func=lambda vacc, one_dose, pop: (vacc + one_dose) / pop,
        sources=[
            f"_{Vaccination.VACCINATED}",
            f"_{Vaccination.ONE_DOSE_ONLY}",
            "_total_population"
        ]
    )

    # Track the rate of adverse events and hospitalisations by age, if adverse events calculations are requested
    if len(params.vaccination.roll_out_components) > 0 and params.vaccination_risk.calculate:
        hospital_sources = []
        request_stratified_output_for_flow(model, "vaccination", AGEGROUP_STRATA, "agegroup")

        for agegroup in AGEGROUP_STRATA:
            model.request_output_for_flow(
                name=f"vaccinationXagegroup{agegroup}",
                flow_name="vaccination",
                source_strata={"agegroup": agegroup},
            )

            # TTS for AstraZeneca vaccines
            model.request_function_output(
                name=f"tts_casesXagegroup_{agegroup}",
                sources=[f"vaccinationXagegroup_{agegroup}"],
                func=lambda vaccinated:
                vaccinated * params.vaccination_risk.tts_rate[agegroup] * params.vaccination_risk.prop_astrazeneca
            )
            model.request_function_output(
                name=f"tts_deathsXagegroup_{agegroup}",
                sources=[f"tts_casesXagegroup_{agegroup}"],
                func=lambda tts_cases:
                tts_cases * params.vaccination_risk.tts_fatality_ratio[agegroup]
            )

            # Myocarditis for mRNA vaccines
            model.request_function_output(
                name=f"myocarditis_casesXagegroup_{agegroup}",
                sources=[f"vaccinationXagegroup_{agegroup}"],
                func=lambda vaccinated:
                vaccinated * params.vaccination_risk.myocarditis_rate[agegroup] * params.vaccination_risk.prop_mrna
            )
            hospital_sources += [
                f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.ICU}",
                f"{PROGRESS}Xagegroup_{agegroup}Xclinical_{Clinical.HOSPITAL_NON_ICU}",
            ]

            # Hospitalisations by age
            hospital_sources_this_age = [s for s in hospital_sources if f"Xagegroup_{agegroup}X" in s]
            model.request_aggregate_output(
                name=f"new_hospital_admissionsXagegroup_{agegroup}",
                sources=hospital_sources_this_age
            )

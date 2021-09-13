from summer import CompartmentalModel

from autumn.models.covid_19.constants import VACCINATION_STRATA, Vaccination
from autumn.models.covid_19.constants import COMPARTMENTS
from autumn.models.covid_19.parameters import Parameters


def request_vaccination_outputs(model: CompartmentalModel, params: Parameters):
    """
    Get the vaccination-related outputs
    """

    # track proportions vaccinated by vaccination status
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

from summer import CompartmentalModel

from autumn.models.covid_19.constants import VACCINATION_STRATA, Vaccination
from autumn.models.covid_19.constants import COMPARTMENTS
from autumn.models.covid_19.parameters import Parameters


def request_common_outputs(model: CompartmentalModel, params: Parameters):

    # Vaccination
    if params.vaccination:
        for stratum in VACCINATION_STRATA:
            model.request_output_for_compartments(
                name=f"{stratum}_number",
                compartments=COMPARTMENTS,
                strata={"vaccination": stratum}
            )
            model.request_function_output(
                name=f"{stratum}_prop",
                func=lambda number, pop: number / pop,
                sources=[f"{stratum}_number", "_total_population"],
            )
        model.request_function_output(
            name="at_least_one_dose_prop",
            func=lambda vacc, one_dose, pop: (vacc + one_dose) / pop,
            sources=[
                f"{Vaccination.VACCINATED}_number",
                f"{Vaccination.ONE_DOSE_ONLY}_number",
                "_total_population"
            ]
        )

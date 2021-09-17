from summer import CompartmentalModel

from autumn.models.covid_19.constants import Compartment, COMPARTMENTS, History
from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA
from autumn.models.covid_19.outputs.standard import request_stratified_output_for_compartment
from autumn.settings import Region


def request_history_outputs(model: CompartmentalModel):
    """
    Proportion seropositive/recovered
    """

    # Note these people are called "naive", but they have actually had past Covid, immunity just hasn't yet waned
    model.request_output_for_compartments(
        name="_recovered",
        compartments=[Compartment.RECOVERED],
        strata={"history": History.NAIVE},
        save_results=False,
    )
    model.request_output_for_compartments(
        name="_experienced",
        compartments=COMPARTMENTS,
        strata={"history": History.EXPERIENCED},
        save_results=False,
    )
    model.request_function_output(
        name="proportion_seropositive",
        sources=["_recovered", "_experienced", "_total_population"],
        func=lambda recovered, experienced, total: (recovered + experienced) / total,
    )

    request_stratified_output_for_compartment(
        model, "_total_population", COMPARTMENTS, AGEGROUP_STRATA, "agegroup", save_results=False
    )
    for agegroup in AGEGROUP_STRATA:
        recovered_name = f"_recoveredXagegroup_{agegroup}"
        total_name = f"_total_populationXagegroup_{agegroup}"
        experienced_name = f"_experiencedXagegroup_{agegroup}"
        model.request_output_for_compartments(
            name=recovered_name,
            compartments=[Compartment.RECOVERED],
            strata={"agegroup": agegroup, "history": History.EXPERIENCED},
            save_results=False,
        )
        model.request_output_for_compartments(
            name=experienced_name,
            compartments=COMPARTMENTS,
            strata={"agegroup": agegroup, "history": History.NAIVE},
            save_results=False,
        )
        model.request_function_output(
            name=f"proportion_seropositiveXagegroup_{agegroup}",
            sources=[recovered_name, experienced_name, total_name],
            func=lambda recovered, experienced, total: (recovered + experienced) / total,
        )


def request_recovered_outputs(model: CompartmentalModel, is_region_vic: bool):
    """
    If not stratified by history status.
    """

    # Unstratified
    model.request_output_for_compartments(
        name="_recovered",
        compartments=[Compartment.RECOVERED],
        save_results=False
    )
    model.request_function_output(
        name="proportion_seropositive",
        sources=["_recovered", "_total_population"],
        func=lambda recovered, total: recovered / total,
    )

    request_stratified_output_for_compartment(
        model, "_total_population", COMPARTMENTS, AGEGROUP_STRATA, "agegroup", save_results=False
    )

    # Stratified by age group
    for agegroup in AGEGROUP_STRATA:
        recovered_name = f"_recoveredXagegroup_{agegroup}"
        total_name = f"_total_populationXagegroup_{agegroup}"
        model.request_output_for_compartments(
            name=recovered_name,
            compartments=[Compartment.RECOVERED],
            strata={"agegroup": agegroup},
            save_results=False,
        )
        model.request_function_output(
            name=f"proportion_seropositiveXagegroup_{agegroup}",
            sources=[recovered_name, total_name],
            func=lambda recovered, total: recovered / total,
        )

    if is_region_vic:
        clusters = [Region.to_filename(region) for region in Region.VICTORIA_SUBREGIONS]
        for cluster in clusters:
            total_name = f"_total_populationXcluster_{cluster}"
            recovered_name = f"_recoveredXcluster_{cluster}"
            model.request_output_for_compartments(
                name=total_name,
                compartments=COMPARTMENTS,
                strata={"cluster": cluster},
                save_results=False,
            )
            model.request_output_for_compartments(
                name=recovered_name,
                compartments=[Compartment.RECOVERED],
                strata={"cluster": cluster},
                save_results=False,
            )
            model.request_function_output(
                name=f"proportion_seropositiveXcluster_{cluster}",
                sources=[recovered_name, total_name],
                func=lambda recovered, total: recovered / total,
            )

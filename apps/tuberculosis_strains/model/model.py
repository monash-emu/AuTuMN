from copy import deepcopy
from summer.model import StratifiedModel
from autumn.constants import Compartment, BirthApproach, Flow
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit.demography import set_model_time_variant_birth_rate
from summer.flow import StandardFlow
from autumn.inputs import get_population_by_agegroup
from autumn.inputs.social_mixing.queries import get_mixing_matrix_specific_agegroups


from apps.tuberculosis_strains.model import preprocess, outputs



def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run a tuberculosis model
    """

    # Define model times.
    time = params["time"]
    integration_times = get_model_times_from_inputs(
        round(time["start"]), time["end"], time["step"], time["critical_ranges"]
    )

    # Define model compartments.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.DETECTED,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
    ]
    infectious_comps = [
        Compartment.INFECTIOUS,
        Compartment.DETECTED,
        Compartment.ON_TREATMENT,
    ]
    infected_comps =[
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.DETECTED,
        Compartment.ON_TREATMENT,
    ]


    # Define initial conditions - 1 infectious person.
    init_conditions = {
        Compartment.EARLY_LATENT: params["initial_early_latent_population"],
        Compartment.LATE_LATENT: params["initial_late_latent_population"],
        Compartment.INFECTIOUS: params["initial_infectious_population"],
        Compartment.DETECTED: params["initial_detected_population"],
        Compartment.ON_TREATMENT: params["initial_on_treatment_population"],
    }


    # Generate derived parameters
    params = preprocess.derived_params.get_derived_params(params)


    # Define inter-compartmental flows.
    flows = deepcopy(preprocess.flows.DEFAULT_FLOWS)

    # Create the model.
    tb_model = StratifiedModel(
        times=integration_times,
        compartment_names=compartments,
        initial_conditions=init_conditions,
        parameters=params,
        requested_flows=flows,
        infectious_compartments=infectious_comps,
        birth_approach=BirthApproach.ADD_CRUDE,
        entry_compartment=Compartment.SUSCEPTIBLE,
        starting_population=int(params["start_population_size"]),
    )

    pop = get_population_by_agegroup(age_breakpoints=params["age_breakpoints"],
                                            country_iso_code=params["iso3"],
                                            year=2000)

    mixing_matrix = get_mixing_matrix_specific_agegroups(country_iso_code=params["iso3"],
                                                         requested_age_breaks=list(map(int, params["age_breakpoints"])),
                                                         time_unit="years")

    # apply age stratification
    age_stratification_name = "age"
    tb_model.stratify(stratification_name=age_stratification_name,
                      strata_request=params["age_breakpoints"],
                      compartments_to_stratify=compartments,
                      comp_split_props=dict(zip(params["age_breakpoints"], [x / sum(pop) for x in pop])),
                      mixing_matrix=mixing_matrix)

    tb_model.time_variants["time_varying_vaccination_coverage"] = lambda t: params["bcg"]["coverage"] if t > params["bcg"]["start_time"] else 0.0
    tb_model.time_variants["time_varying_unvaccinated_coverage"] = lambda t: 1 - params["bcg"]["coverage"] if t > params["bcg"]["start_time"] else 1.0
    #
    # def new_time_varying_fcn(t):
    #     return t**2
    #
    # tb_model.time_variants["death_rate"] = new_time_varying_fcn


    # apply vaccination stratification
    vac_stratification_name = "vac"
    vac_strata_requested = ["unvaccinated", "vaccinated"]
    vaccination_flow_adjustments = dict(zip(["contact_rate" + "X" + age_stratification_name + "_" + age_group for age_group in params["age_breakpoints"]],
                                            [{"unvaccinated": 1.0, "vaccinated": params["bcg"]["rr_infection_vaccinated"]} for _ in range(len(params["age_breakpoints"]))]))

    vaccination_flow_adjustments["crude_birth_rateXage_0"] = {"unvaccinated": "time_varying_unvaccinated_coverage", "vaccinated": "time_varying_vaccination_coverage"}

    tb_model.stratify(stratification_name=vac_stratification_name,
                      strata_request=vac_strata_requested,
                      compartments_to_stratify=[Compartment.SUSCEPTIBLE],
                      comp_split_props={"unvaccinated": 1.0, "vaccinated": 0.0},
                      flow_adjustments=vaccination_flow_adjustments,
                      )



    # apply organ stratification
    organ_stratification_name = "organ"
    organ_strata_requested = ["smear_positive", "smear_negative", "extra_pulmonary"]
    organ_flow_adjustments = dict(zip(["early_activation_rate" + "X" + age_stratification_name + "_" + age_group
                                            for age_group in params["age_breakpoints"]],
                                      [{"smear_positive": params["organ"]["sp_prop"],
                                        "smear_negative": params["organ"]["sn_prop"],
                                        "extra_pulmonary": params["organ"]["e_prop"]} for _ in range(len(params["age_breakpoints"]))]))


    tb_model.stratify(stratification_name=organ_stratification_name,
                      strata_request=organ_strata_requested,
                      compartments_to_stratify=infectious_comps,
                      comp_split_props={"smear_positive": params["organ"]["sp_prop"], "smear_negative": params["organ"]["sn_prop"], "extra_pulmonary": params["organ"]["e_prop"]},
                      flow_adjustments=organ_flow_adjustments,
                      # flow_adjustments={"early_activation_rateXstrain_ds": {"smear_positive": 0.25, "smear_negative": 0.25, "extra_pulmonary": 0.5},
                      #                   "early_activation_rateXstrain_mdr": {"smear_positive": 0.25, "smear_negative": 0.25, "extra_pulmonary": 0.5}},
                      infectiousness_adjustments={"smear_positive": params["organ"]["sp_foi"], "smear_negative": params["organ"]["sn_foi"], "extra_pulmonary": params["organ"]["e_foi"]},

                      )

    # apply strain stratification
    # tb_model.stratify(stratification_name="strain",
    #                   strata_request=["ds", "mdr"],
    #                   compartments_to_stratify=infected_comps,
    #                   comp_split_props={"ds": 0.5, "mdr": 0.5},
    #                   flow_adjustments={"infect_death_rate": {"ds": 1.0, "mdr": 2.0},
    #                                     "early_activation_rate": {"ds": 1.0, "mdr": 0.5}},
    #                   infectiousness_adjustments={"ds": 1.0, "mdr": 0.8})



    # Register derived output functions, which are calculations based on the model's compartment values or flows.
    # These are calculated after the model is run.
    outputs.get_all_derived_output_functions(
        params["calculated_outputs"], params["outputs_stratification"], tb_model
    )

    return tb_model

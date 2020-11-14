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
    age_flow_adjustments = {"stabilisation_rate": {"0": params["stabilisation_rate_stratified"]["age_0"],
                                                   "5": params["stabilisation_rate_stratified"]["age_5"],
                                                   "15": params["stabilisation_rate_stratified"]["age_15"]},
                            "late_activation_rate": {"0": params["late_activation_rate_stratified"]["age"]["age_0"],
                                                     "5": params["late_activation_rate_stratified"]["age"]["age_5"],
                                                     "15": params["late_activation_rate_stratified"]["age"]["age_15"]}}
    tb_model.stratify(stratification_name=age_stratification_name,
                      strata_request=params["age_breakpoints"],
                      compartments_to_stratify=compartments,
                      comp_split_props=dict(zip(params["age_breakpoints"], [x / sum(pop) for x in pop])),
                      flow_adjustments=age_flow_adjustments,
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
    organ_flow_adjustments = dict()
    organ_flow_adjustments.update(dict(zip(["early_activation_rate" + "X" + age_stratification_name + "_" + age_group
                                            for age_group in params["age_breakpoints"]],
                                      [{"smear_positive": params["organ"]["sp_prop"],
                                        "smear_negative": params["organ"]["sn_prop"],
                                        "extra_pulmonary": params["organ"]["e_prop"]} for _ in range(len(params["age_breakpoints"]))]))
                                  )
    organ_flow_adjustments.update(dict(zip(["late_activation_rate" + "X" + age_stratification_name + "_" + age_group
                                            for age_group in params["age_breakpoints"]],
                                      [{"smear_positive": params["organ"]["sp_prop"],
                                        "smear_negative": params["organ"]["sn_prop"],
                                        "extra_pulmonary": params["organ"]["e_prop"]} for _ in range(len(params["age_breakpoints"]))]))
                                  )

    organ_flow_adjustments.update(dict(zip(["detection_rate" + "X" + age_stratification_name + "_" + age_group
                                            for age_group in params["age_breakpoints"]],
                                      [{"smear_positive": params["detection_rate_stratified"]["organ"]["sp"],
                                        "smear_negative": params["detection_rate_stratified"]["organ"]["sn"],
                                        "extra_pulmonary": params["detection_rate_stratified"]["organ"]["e"]} for _ in range(len(params["age_breakpoints"]))]))
                                  )

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
    strain_stratification_name = "strain"
    strain_strata_requested = ["ds", "mdr"]
    strain_flow_adjustments = dict()
    strain_flow_adjustments.update(dict(zip(["treatment_commencement_rate" + "X" + age_stratification_name + "_" + age_group +
                                                "X" + organ_stratification_name + "_" + organ
                                            for age_group in params["age_breakpoints"]
                                                for organ in organ_strata_requested],
                                      [{"ds": params["treatment_commencement_rate_stratified"]["strain"]["ds"],
                                        "mdr": params["treatment_commencement_rate_stratified"]["strain"]["mdr"]} for _ in range(len(params["age_breakpoints"]) * len(organ_strata_requested))]))
                                  )
    strain_flow_adjustments.update(
        dict(zip(["treatment_recovery_rate" + "X" + age_stratification_name + "_" + age_group +
                  "X" + organ_stratification_name + "_" + organ
                  for age_group in params["age_breakpoints"]
                  for organ in organ_strata_requested],
                 [{"ds": params["treatment_recovery_rate_stratified"]["strain"]["ds"],
                   "mdr": params["treatment_recovery_rate_stratified"]["strain"]["mdr"]} for _ in
                  range(len(params["age_breakpoints"]) * len(organ_strata_requested))]))
        )
    strain_flow_adjustments.update(
        dict(zip(["treatment_death_rate" + "X" + age_stratification_name + "_" + age_group +
                  "X" + organ_stratification_name + "_" + organ
                  for age_group in params["age_breakpoints"]
                  for organ in organ_strata_requested],
                 [{"ds": params["treatment_death_rate_stratified"]["strain"]["ds"],
                   "mdr": params["treatment_death_rate_stratified"]["strain"]["mdr"]} for _ in
                  range(len(params["age_breakpoints"]) * len(organ_strata_requested))]))
    )
    strain_flow_adjustments.update(
        dict(zip(["treatment_default_rate" + "X" + age_stratification_name + "_" + age_group +
                  "X" + organ_stratification_name + "_" + organ
                  for age_group in params["age_breakpoints"]
                  for organ in organ_strata_requested],
                 [{"ds": params["treatment_default_rate_stratified"]["strain"]["ds"],
                   "mdr": params["treatment_default_rate_stratified"]["strain"]["mdr"]} for _ in
                  range(len(params["age_breakpoints"]) * len(organ_strata_requested))]))
    )

    tb_model.stratify(stratification_name=strain_stratification_name,
                      strata_request=strain_strata_requested,
                      compartments_to_stratify=infected_comps,
                      comp_split_props={"ds": 0.5, "mdr": 0.5},
                      flow_adjustments=strain_flow_adjustments,
    #                   flow_adjustments={"infect_death_rate": {"ds": 1.0, "mdr": 2.0},
    #                                     "early_activation_rate": {"ds": 1.0, "mdr": 0.5}},
                      infectiousness_adjustments={"ds": 1.0, "mdr": 0.8})


    # apply classification stratification
    # classification_stratification_name = "classified"
    # classification_strata_requested = ["correctly", "incorrectly"]

    # tb_model.stratify(stratification_name=classification_stratification_name,
    #                   strata_request=classification_strata_requested,
    #                   compartments_to_stratify=[Compartment.DETECTED, Compartment.ON_TREATMENT])


    # apply retention stratification
    retention_stratification_name = "retained"
    retention_strata_requested = ["yes", "no"]

    retention_flow_adjustments = dict()
    retention_flow_adjustments.update(dict(zip(["detection_rate" + "X" + age_stratification_name + "_" + age_group +
                                                "X" + organ_stratification_name + "_" + organ +
                                                "X" + strain_stratification_name + "_" + strain
                                            for age_group in params["age_breakpoints"]
                                                for organ in organ_strata_requested
                                                    for strain in strain_strata_requested],
                                      [{"yes": params["retention_rate"],
                                        "no": 1 - params["retention_rate"]} for _ in range(len(params["age_breakpoints"]) * len(organ_strata_requested) * len(strain_strata_requested))]))
                                  )

    retention_flow_adjustments.update(dict(zip(["missed_to_active_rate" + "X" + age_stratification_name + "_" + age_group +
                                                "X" + organ_stratification_name + "_" + organ +
                                                "X" + strain_stratification_name + "_" + strain
                                                for age_group in params["age_breakpoints"]
                                                for organ in organ_strata_requested
                                                for strain in strain_strata_requested],
                                               [{"yes": 0.0,
                                                 "no": 1.0} for _ in range(
                                                   len(params["age_breakpoints"]) * len(organ_strata_requested) * len(
                                                       strain_strata_requested))]))
                                      )

    tb_model.stratify(stratification_name=retention_stratification_name,
                      strata_request=retention_strata_requested,
                      compartments_to_stratify=[Compartment.DETECTED],
                      flow_adjustments=retention_flow_adjustments)


    # Register derived output functions, which are calculations based on the model's compartment values or flows.
    # These are calculated after the model is run.
    outputs.get_all_derived_output_functions(
        params["calculated_outputs"], params["outputs_stratification"], tb_model
    )

    return tb_model

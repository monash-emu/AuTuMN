import pylatex as pl
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from jax import numpy as jnp

from summer2 import CompartmentalModel, Stratification, StrainStratification
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from autumn.projects.austcovid.doc_utils import TextElement, FigElement, DocumentedProcess


REF_DATE = datetime(2019, 12, 31)
BASE_PATH = Path(__file__).parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


def make_voc_seed_func(entry_rate: float, start_time: float, seed_duration: float):
    def voc_seed_func(time, entry_rate, start_time, seed_duration):
        offset = time - start_time
        return jnp.where(offset > 0, jnp.where(offset < seed_duration, entry_rate, 0.0), 0.0)
    return Function(voc_seed_func, [Time, entry_rate, start_time, seed_duration])


class DocumentedAustModel(DocumentedProcess):

    def __init__(self, doc=None, add_documentation=False):
        super().__init__(doc, add_documentation)

    def build_base_model(
        self,
        start_date: datetime,
        end_date: datetime,
        compartments: list,
    ):
        """
        Build the base model object with no features yet, as described below.
        """

        self.model = CompartmentalModel(
            times=(
                (start_date - REF_DATE).days, 
                (end_date - REF_DATE).days,
            ),
            compartments=compartments,
            infectious_compartments=("infectious",),
            ref_date=REF_DATE,
        )

        if self.add_documentation:
            description = "The base model consists of three states, " \
                "representing fully susceptible, infected (and infectious) and recovered persons. " \
                f"The model is run from {str(start_date.date())} to {str(end_date.date())}. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def set_model_starting_conditions(self):
        """
        Add the starting populations to the model as described below.
        """

        population = 2.6e7
        self.model.set_initial_population({"susceptible": population})
        
        if self.add_documentation:
            description = f"The simulation starts with {str(population / 1e6)} million susceptible persons only, " \
                "with infectious persons only introduced through strain seeding as described below. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_infection_to_model(self):
        """
        Add the infection process as described below.
        """

        process = "infection"
        origin = "susceptible"
        destination = "infectious"
        self.model.add_infection_frequency_flow(
            process,
            Parameter("contact_rate"),
            origin,
            destination,
        )
        
        if self.add_documentation:
            description = f"The {process} moves people from the {origin} " \
                f"compartment to the {destination} compartment, " \
                "under the frequency-dependent transmission assumption. "
            self.add_element_to_doc("General model construction", TextElement(description))
            
    def add_recovery_to_model(self):
        """
        Add recovery as described below.
        """

        process = "recovery"
        origin = "infectious"
        destination = "recovered"
        self.model.add_transition_flow(process, 1.0 / Parameter("infectious_period"), origin, destination)

        if self.add_documentation:
            description = f"The {process} process moves " \
                f"people directly from the {origin} state to the {destination} compartment, " \
                "with the rate of transition calculated as the reciprocal of the infectious period."
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_notifications_output_to_model(self):
        """
        Track notifications as described below.
        """

        process = "onset"
        output = "notifications"
        transition = "infection"
        self.model.request_output_for_flow(process, transition, save_results=False)
        self.model.request_function_output(output, func=DerivedOutput(process) * Parameter("cdr"))

        if self.add_documentation:
            description = f"Modelled {output} are calculated as " \
                f"the absolute rate of {transition} in the community " \
                "multiplied by the case detection rate. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def build_polymod_britain_matrix(
        self,
        strata: list,
    ) -> np.array:
        """
        Get the raw data for Great Britain as described below.

        Args:
            strata: The strata to apply in age stratification
        Returns:
            15 by 15 matrix with daily contact rates for age groups
        """

        values = [
            [1.92, 0.65, 0.41, 0.24, 0.46, 0.73, 0.67, 0.83, 0.24, 0.22, 0.36, 0.20, 0.20, 0.26, 0.13],
            [0.95, 6.64, 1.09, 0.73, 0.61, 0.75, 0.95, 1.39, 0.90, 0.16, 0.30, 0.22, 0.50, 0.48, 0.20],
            [0.48, 1.31, 6.85, 1.52, 0.27, 0.31, 0.48, 0.76, 1.00, 0.69, 0.32, 0.44, 0.27, 0.41, 0.33],
            [0.33, 0.34, 1.03, 6.71, 1.58, 0.73, 0.42, 0.56, 0.85, 1.16, 0.70, 0.30, 0.20, 0.48, 0.63],
            [0.45, 0.30, 0.22, 0.93, 2.59, 1.49, 0.75, 0.63, 0.77, 0.87, 0.88, 0.61, 0.53, 0.37, 0.33],
            [0.79, 0.66, 0.44, 0.74, 1.29, 1.83, 0.97, 0.71, 0.74, 0.85, 0.88, 0.87, 0.67, 0.74, 0.33],
            [0.97, 1.07, 0.62, 0.50, 0.88, 1.19, 1.67, 0.89, 1.02, 0.91, 0.92, 0.61, 0.76, 0.63, 0.27],
            [1.02, 0.98, 1.26, 1.09, 0.76, 0.95, 1.53, 1.50, 1.32, 1.09, 0.83, 0.69, 1.02, 0.96, 0.20],
            [0.55, 1.00, 1.14, 0.94, 0.73, 0.88, 0.82, 1.23, 1.35, 1.27, 0.89, 0.67, 0.94, 0.81, 0.80],
            [0.29, 0.54, 0.57, 0.77, 0.97, 0.93, 0.57, 0.80, 1.32, 1.87, 0.61, 0.80, 0.61, 0.59, 0.57],
            [0.33, 0.38, 0.40, 0.41, 0.44, 0.85, 0.60, 0.61, 0.71, 0.95, 0.74, 1.06, 0.59, 0.56, 0.57],
            [0.31, 0.21, 0.25, 0.33, 0.39, 0.53, 0.68, 0.53, 0.55, 0.51, 0.82, 1.17, 0.85, 0.85, 0.33],
            [0.26, 0.25, 0.19, 0.24, 0.19, 0.34, 0.40, 0.39, 0.47, 0.55, 0.41, 0.78, 0.65, 0.85, 0.57],
            [0.09, 0.11, 0.12, 0.20, 0.19, 0.22, 0.13, 0.30, 0.23, 0.13, 0.21, 0.28, 0.36, 0.70, 0.60],
            [0.14, 0.15, 0.21, 0.10, 0.24, 0.17, 0.15, 0.41, 0.50, 0.71, 0.53, 0.76, 0.47, 0.74, 1.47],
        ]

        matrix = np.array(values).T  # Transpose

        if self.add_documentation:
            description = "We took unadjusted estimates for interpersonal rates of contact by age " \
                "from the United Kingdom data provided by Mossong et al.'s POLYMOD study \cite{mossong2008}. " \
                "The data were obtained from https://doi.org/10.1371/journal.pmed.0050074.st005 " \
                "on 12th February 2023 (downloaded in their native docx format). " \
                "The matrix is transposed because summer assumes that rows represent infectees " \
                "and columns represent infectors, whereas the POLYMOD data are labelled " \
                "`age of contact' for the rows and `age group of participant' for the columns."
            self.add_element_to_doc("General model construction", TextElement(description))
            filename = "raw_matrix.jpg"
            matrix_plotly_fig = px.imshow(matrix, x=strata, y=strata)
            matrix_plotly_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Raw matrices from Great Britain POLYMOD. Values are contacts per person per day."
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        return matrix

    def adapt_gb_matrix_to_aust(
        self,
        unadjusted_matrix: np.array, 
        strata: list, 
    ) -> np.array:
        """
        Adjust the Great Britain matrix to Australia's population distribution,
        as described below.

        Args:
            unadjusted_matrix: The unadjusted matrix
            strata: The strata to apply in age stratification
        Returns:
            Matrix adjusted to target population
        """
        
        # UK population distributions
        uk_pops_list = [
            3458060, 3556024, 3824317, 3960916, 3911291, 3762213, 4174675, 4695853, 
            4653082, 3986098, 3620216, 3892985, 3124676, 2706365, 6961183,
        ]
        uk_age_pops = pd.Series(uk_pops_list, index=strata)
        uk_age_props = uk_age_pops / uk_age_pops.sum()
        
        # Australian distributions from https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/jun-2022/31010do002_202206.xlsx, 13/2/23
        aust_percs_list = [
            5.8, 6.2, 6.3, 5.9, 6.3, 7.0, 7.3, 7.3, 6.6, 6.2, 6.4, 5.9, 5.7, 5.0, 4.4, 3.4, 2.2, 2.1,
        ]
        aust_percs_list = aust_percs_list[:14] + [sum(aust_percs_list[14:])]  # Adapt to our age groups
        aust_age_percs = pd.Series(aust_percs_list, index=strata)
        aust_age_props = aust_age_percs / aust_age_percs.sum()  # Sum is just 100
        
        # Calculation
        aust_uk_ratios = aust_age_props / uk_age_props
        adjusted_matrix = np.dot(unadjusted_matrix, np.diag(aust_uk_ratios))
        
        if self.add_documentation:
            description = "Matrices were adjusted to account for the differences in the age distribution of the " \
                "Australian population distribution in 2022 compared to the population of Great Britain in 2008. " \
                "The matrices were adjusted by taking the dot product of the unadjusted matrices and the diagonal matrix " \
                "containing the vector of the ratios between the proportion of the British and Australian populations " \
                "within each age bracket as its diagonal elements. "
            self.add_element_to_doc("Age stratification", TextElement(description))
            filename = "adjusted_matrix.jpg"
            matrix_plotly_fig = px.imshow(unadjusted_matrix, x=strata, y=strata)
            matrix_plotly_fig.write_image(SUPPLEMENT_PATH / filename)
            caption = "Matrices adjusted to Australian population. Values are contacts per person per day."
            self.add_element_to_doc("Age stratification", FigElement(filename, caption=caption))

        return adjusted_matrix

    def add_age_stratification_to_model(
        self,
        compartments: list,
        strata: list,
        matrix: np.array,
    ):
        """
        Add age stratification to the model as described below,
        using summer's Stratification class rather than AgeStratification
        because we are not requesting ageing between age brackets.

        Args:
            compartments: All the unstratified model compartments
            strata: The strata to apply
            matrix: The mixing matrix to apply
        """

        age_strat = Stratification("agegroup", strata, compartments)
        age_strat.set_mixing_matrix(matrix)
        self.model.stratify_with(age_strat)

        if self.add_documentation:
            description = "We stratified all compartments of the base model " \
                "into sequential age brackets in five year " \
                "bands from age 0 to 4 through to age 65 to 69 " \
                "with a final age band to represent those aged 70 and above. " \
                "These age brackets were chosen to match those used by the POLYMOD survey. "
            self.add_element_to_doc("Age stratification", TextElement(description))

    def get_strain_stratification(self):
        """
        Add strain stratification to the model as described below.
        """

        # The strains we're working with
        all_strains = ["ba1", "ba2"]
        starting_strain = all_strains[0]  # BA.1
        other_strains = all_strains[1:]  # The others, currently just BA.2

        # The stratification object
        strain_strat = StrainStratification("strain", all_strains, ["infectious"])

        # The starting population split
        population_split = {starting_strain: 1.0}
        population_split.update({strain: 0.0 for strain in other_strains})
        strain_strat.set_population_split(population_split)

        if self.add_documentation:
            description = "We stratified the infectious compartment according to strain, " \
                "including compartments to represent strain BA.1 and BA.2. " \
                "This was implemented using summer's `StrainStratication' class. " \
                "All of the starting infectious seed was assigned to the BA.1 category. "
            self.add_element_to_doc("Strain stratification", TextElement(description))

        return strain_strat, starting_strain, other_strains

    def adjust_strain_infectiousness(self, strat, starting_strain, other_strains):
        """
        Adjust the infectiousness of the SARS-CoV-2 sub-variants modelled.
        """
        
        infectiousness_adjs = {starting_strain: None}
        infectiousness_adjs.update({strain: Parameter(f"{strain}_rel_infness") for strain in other_strains})
        strat.set_flow_adjustments("infection", infectiousness_adjs)

        if self.add_documentation:
            description = "The relative infectiousness of the BA.2 strain was adjusted relative " \
                "to the starting strain (BA.1) as indicated in the parameters table. "
            self.add_element_to_doc("Strain stratification", TextElement(description))
        
        return strat

    def seed_vocs(self):

        for strain in self.model.stratifications["strain"].strata:
            voc_seed_func = make_voc_seed_func(
                Parameter("seed_rate"), 
                Parameter(f"{strain}_seed_time"), 
                Parameter("seed_duration")
            )
            self.model.add_importation_flow(
                "seed_{strain}",
                voc_seed_func,
                "infectious",
                dest_strata={"strain": strain},
                split_imports=True,
            )


def build_aust_model(
    start_date: datetime,
    end_date: datetime,
    doc: pl.document.Document,
    add_documentation: bool=False,
) -> CompartmentalModel:
    """
    Build a fairly basic model, as described in the component functions called.
    
    Returns:
        The model object
    """

    # Basic model construction
    compartments = [
        "susceptible",
        "infectious",
        "recovered",
    ]
    aust_model = DocumentedAustModel(doc, add_documentation)
    aust_model.build_base_model(start_date, end_date, compartments)
    aust_model.set_model_starting_conditions()
    aust_model.add_infection_to_model()
    aust_model.add_recovery_to_model()
    aust_model.add_notifications_output_to_model()

    # Age stratification
    age_strata = list(range(0, 75, 5))
    matrix = aust_model.build_polymod_britain_matrix(age_strata)
    adjusted_matrix = aust_model.adapt_gb_matrix_to_aust(matrix, age_strata)
    aust_model.add_age_stratification_to_model(compartments, age_strata, adjusted_matrix)
    
    # Strain stratification
    strain_strat, starting_strain, other_strains = aust_model.get_strain_stratification()
    aust_model.adjust_strain_infectiousness(strain_strat, starting_strain, other_strains)
    aust_model.model.stratify_with(strain_strat)
    aust_model.seed_vocs()

    # Documentation
    if add_documentation:
        aust_model.compile_doc()
    return aust_model.model

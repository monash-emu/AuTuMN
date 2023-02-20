import pylatex as pl
from pylatex.section import Section
from pylatex.utils import NoEscape
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from summer2 import CompartmentalModel, Stratification, StrainStratification
from summer2.parameters import Parameter, DerivedOutput

REF_DATE = datetime(2019, 12, 31)
BASE_PATH = Path(__file__).parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


class DocElement:
    def __init__():
        pass

    def emit_latex():
        pass


class TextElement(DocElement):
    """
    Write text input to TeX document using PyLaTeX commands.
    """
    def __init__(
            self, 
            text: str,
        ):
        """
        Set up object with text input.

        Args:
            text: The text to write
        """
        self.text = NoEscape(text) if "\cite{" in text else text

    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the text to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        doc.append(self.text)


class FigElement(DocElement):
    """
    Add a figure to a TeX document using PyLaTeX commands.
    """
    def __init__(
            self, 
            fig_name: str,
            caption: str="",
            resolution: str="350px",
        ):
        """
        Set up object with figure input and other requests.

        Args:
            fig_name: The name of the figure to write
            caption: Figure caption
            resolution: Resolution to write to
        """
        self.fig_name = fig_name
        self.caption = caption
        self.resolution = resolution
    
    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the figure to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        with doc.create(pl.Figure()) as plot:
            plot.add_image(self.fig_name, width=self.resolution)
            plot.add_caption(self.caption)


class DocumentedModel:

    def __init__(self, doc=None, add_documentation=False):
        self.doc = doc
        self.add_documentation = add_documentation
        self.doc_sections = {}

    def add_element_to_doc(
        self, 
        section_name: str, 
        element: DocElement,
    ):
        """
        Add a new element to the list of elements to be included in a document section.

        Args:
            section_name: Name of the section to add to
            element: The object to include in the section
        """
        if section_name not in self.doc_sections:
            self.doc_sections[section_name] = []
        self.doc_sections[section_name].append(element)

    def compile_doc(self):
        """
        Apply all the document elements to the document,
        looping through each section and using each element's emit_latex method.
        """
        for section in self.doc_sections:
            with self.doc.create(Section(section)):
                for element in self.doc_sections[section]:
                    element.emit_latex(self.doc)


class DocumentedAustModel(DocumentedModel):

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
                "representing fully susceptible, infected (and infectious) and recovered persons. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def set_model_starting_conditions(self):
        """
        Add the starting populations to the model as described below.
        """

        self.model.set_initial_population(
            {
                "susceptible": 2.6e7,
                "infectious": 1.0,
            }
        )
        
        if self.add_documentation:
            description = "The simulation starts with 26 million susceptible persons " \
                "and one infectious person to seed the epidemic. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_infection_to_model(self):
        """
        Add infection as described below.
        """

        self.model.add_infection_frequency_flow(
            "infection",
            Parameter("contact_rate"),
            "susceptible",
            "infectious",
        )
        
        if self.add_documentation:
            description = "Infection moves people from the fully susceptible " \
                "compartment to the infectious compartment, " \
                "under the frequency-dependent transmission assumption. "
            self.add_element_to_doc("General model construction", TextElement(description))
            
    def add_recovery_to_model(self):
        """
        Add recovery as described below.
        """

        self.model.add_transition_flow(
            "recovery",
            1.0 / Parameter("infectious_period"),
            "infectious",
            "recovered",
        )

        if self.add_documentation:
            description = "The process recovery process moves " \
                "people directly from the infectious state to a recovered compartment. "
            self.add_element_to_doc("General model construction", TextElement(description))

    def add_notifications_output_to_model(self):
        """
        Track notifications as described below.
        """

        self.model.request_output_for_flow(
            "onset",
            "infection",
            save_results=False,
        )
        self.model.request_function_output(
            "notifications",
            func=DerivedOutput("onset") * Parameter("cdr"),
        )

        if self.add_documentation:
            description = "Notifications are calculated as " \
                "the absolute rate of infection in the community " \
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
            location = "raw_matrix.jpg"
            matrix_plotly_fig = px.imshow(matrix, x=strata, y=strata)
            matrix_plotly_fig.write_image(SUPPLEMENT_PATH / location)
            caption = "Raw matrices from Great Britain POLYMOD. Values are contacts per person per day."
            self.add_element_to_doc("Age stratification", FigElement(location, caption=caption))

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
            location = "adjusted_matrix.jpg"
            matrix_plotly_fig = px.imshow(unadjusted_matrix, x=strata, y=strata)
            matrix_plotly_fig.write_image(SUPPLEMENT_PATH / location)
            caption = "Matrices adjusted to Australian population. Values are contacts per person per day."
            self.add_element_to_doc("Age stratification", FigElement(location, caption=caption))

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
        age_strat = Stratification(
            "agegroup", 
            strata, 
            compartments,
        )
        age_strat.set_mixing_matrix(matrix)
        self.model.stratify_with(age_strat)

        if self.add_documentation:
            description = "We stratified all compartments of the base model " \
                "into sequential age brackets in five year " \
                "bands from age 0 to 4 through to age 65 to 69 " \
                "with a final age band to represent those aged 70 and above. " \
                "These age brackets were chosen to match those used by the POLYMOD survey. "
            self.add_element_to_doc("Age stratification", TextElement(description))

    def add_strain_stratification_to_model(self):
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

        # Infectiousness
        infectiousness_adjs = {starting_strain: None}
        infectiousness_adjs.update({strain: Parameter(f"{strain}_rel_infness") for strain in other_strains})
        strain_strat.set_flow_adjustments("infection", infectiousness_adjs)

        # Apply the stratification
        self.model.stratify_with(strain_strat)

        if self.add_documentation:
            description = "We stratified the infectious compartment according to strain, " \
                "including compartments to represent strain BA.1 and BA.2. " \
                "This was implemented using summer's `StrainStratication' class. " \
                "All of the starting infectious seed was assigned to the BA.1 category, " \
                "and no parameters were modified to distinguish these two strains. " \
                "The relative infectiousness of each strain is adjusted relative " \
                "to the starting strain (BA.1) as indicated in the parameters table. "
            self.add_element_to_doc("Strain stratification", TextElement(description))

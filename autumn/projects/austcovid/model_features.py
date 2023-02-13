import pylatex as pl
from pylatex.section import Section
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

from summer2 import CompartmentalModel, Stratification
from summer2.parameters import Parameter, DerivedOutput

REF_DATE = datetime(2019, 12, 31)


def build_base_model(
    start_date: datetime,
    end_date: datetime,
    doc: pl.document.Document,
):
    """
    Generate the base compartmental model.

    Args:
        model: The model object
        doc: The description document
    """

    model = CompartmentalModel(
        times=(
            (start_date - REF_DATE).days, 
            (end_date - REF_DATE).days,
        ),
        compartments=(
            "susceptible",
            "infectious",
            "recovered",
        ),
        infectious_compartments=("infectious",),
        ref_date=REF_DATE,
    )
    
    description = "The base model consists of just three states, " \
        "representing fully susceptible, infected (and infectious) and recovered persons. "

    if isinstance(doc, pl.document.Document):
        with doc.create(Section("General model construction")):
            doc.append(description)

    return model


def set_model_starting_conditions(
    model: CompartmentalModel,
    doc: pl.document.Document,
):
    """
    Add the starting populations to the model as described below.

    Args:
        model: The model object
        doc: The description document
    """

    model.set_initial_population(
        {
            "susceptible": 2.6e7,
            "infectious": 1.0,
        }
    )
    
    description = "The simulation starts with 26 million susceptible persons " \
        "and one infectious person to seed the epidemic. "

    if isinstance(doc, pl.document.Document):
        doc.append(description)


def add_infection_to_model(
    model: CompartmentalModel,
    doc: pl.document.Document,
):
    """
    Add infection as described below.

    Args:
        model: The model object
        doc: The description document
    """

    model.add_infection_frequency_flow(
        "infection",
        Parameter("contact_rate"),
        "susceptible",
        "infectious",
    )
    
    description = "Infection moves people from the fully susceptible " \
        "compartment to the infectious compartment, " \
        "under the frequency-dependent transmission assumption. "

    if isinstance(doc, pl.document.Document):
        doc.append(description)


def add_recovery_to_model(
    model: CompartmentalModel,
    doc: pl.document.Document,
):  
    """
    Add recovery as described below.

    Args:
        model: The model object
        doc: The description document
    """

    model.add_transition_flow(
        "recovery",
        1.0 / Parameter("infectious_period"),
        "infectious",
        "recovered",
    )

    description = "The process recovery process moves " \
        "people directly from the infectious state to a recovered compartment. "

    if isinstance(doc, pl.document.Document):
        doc.append(description)


def add_notifications_output_to_model(
    model: CompartmentalModel,
    doc: pl.document.Document,
):
    """
    Track notifications as described below.

    Args:
        model: The model object
        doc: The description document
    """

    model.request_output_for_flow(
        "onset",
        "infection",
        save_results=False,
    )
    model.request_function_output(
        "notifications",
        func=DerivedOutput("onset") * Parameter("cdr"),
    )

    description = "Notifications are calculated as " \
        "the absolute rate of infection in the community " \
        "multiplied by the case detection rate. "

    if isinstance(doc, pl.document.Document):
        doc.append(description)


def add_age_stratification_to_model(
    model: CompartmentalModel,
    doc: pl.document.Document,
):
    """
    Add age stratification to the model as described below,
    using summer's Stratification class rather than AgeStratification
    because we are not requesting ageing between age brackets.

    Args:
        model: The model object
        doc: The description document
    """
    age_strat = Stratification(
        "agegroup", 
        range(0, 75, 5), 
        model.compartments,
    )
    model.stratify_with(age_strat)

    description = "We stratified all compartments of the base model " \
        "into sequential age brackets in five year " \
        "bands from age 0 to 4 through to age 65 to 69 " \
        "with a final age band to represent those aged 70 and above. " \
        "These age brackets were chosen to match those used by the POLYMOD survey. "

    if isinstance(doc, pl.document.Document):
        with doc.create(Section("Age stratification")):
            doc.append(description)


def build_polymod_britain_matrix(
    model: CompartmentalModel,
    doc: pl.document.Document,
) -> np.array:
    """
    Get the raw data for Great Britain as described below.

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

    description = "We took unadjusted estimates for interpersonal rates of contact by age " \
        "from the United Kingdom data provided by Mossong et al.'s POLYMOD study. " \
        "The data were obtained from https://doi.org/10.1371/journal.pmed.0050074.st005 " \
        "on 12th February 2023, downloaded in their native docx format and transcribed here. " \
        "The matrix is transposed because summer assumes that rows represent infectees " \
        "and columns represent infectors, whereas the POLYMOD data are labelled " \
        "'age of contact' for the rows and 'age group of participant' for the columns."

    if isinstance(doc, pl.document.Document):
        doc.append(description)
        age_strata = model.stratifications["agegroup"].strata
        matrix_plotly_fig = px.imshow(matrix, x=age_strata, y=age_strata)
        matrix_plotly_fig.write_image("supplement/raw_matrix.jpg")
        with doc.create(pl.Figure()) as plot:
            plot.add_image("raw_matrix.jpg", width="350px")
            plot.add_caption("Raw matrices from Great Britain POLYMOD.")

    return matrix

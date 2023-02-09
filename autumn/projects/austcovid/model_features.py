import pylatex as pl
from datetime import datetime

from summer2 import CompartmentalModel
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

    if type(doc) == pl.document.Document:
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

    if type(doc) == pl.document.Document:
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

    if type(doc) == pl.document.Document:
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

    if type(doc) == pl.document.Document:
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

    if type(doc) == pl.document.Document:
        doc.append(description)

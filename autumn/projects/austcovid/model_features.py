import pylatex as pl

from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


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
        "multiplied by the case detection rate."

    if type(doc) == pl.document.Document:
        doc.append(description)

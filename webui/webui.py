#!/usr/bin/env python

from flask import Flask
from flask import render_template
from flask import redirect
from flask import url_for

from flask_wtf import FlaskForm

from wtforms import StringField
from wtforms import SelectField
from wtforms import SubmitField
from wtforms import BooleanField
from wtforms import IntegerField
from wtforms import DecimalField
from wtforms.fields.html5 import DecimalRangeField


COUNTRIES = [
    "Afghanistan",
    "Albania",
    "Angola",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahrain",
    "Bangladesh",
    "Belarus",
    "Belgium",
    "Benin",
    "Bhutan",
    "Botswana",
    "Brazil",
    "Bulgaria",
    "Burundi",
    "Cameroon",
    "Chad",
    "Chile",
    "Croatia",
    "Djibouti",
    "Ecuador",
    "Estonia",
    "Ethiopia",
    "Fiji",
    "Gabon",
    "Georgia",
    "Ghana",
    "Guatemala",
    "Guinea",
    "Philippines",
    "Romania",
]
DEFAULT_COUNTRY = "Philippines"


INTEGRATION_METHODS = [
    "Runge Kutta",
    "Explicit",
]
DEFAULT_INTEGRATION_METHOD = "Explicit"


FITTING_METHODS = [
    "Method 1",
    "Method 2",
    "Method 3",
    "Method 4",
    "Method 5",
]
DEFAULT_FITTING_METHOD = "Method 5"


N_ORGANS = [
    "Pos / Neg / Extra",
    "Pos / Neg",
    "Unstratified",
]
DEFAULT_N_ORGAN = "Pos / Neg / Extra"


N_STRAINS = [
    "Single strain",
    "DS / MDR",
    "DS / MDR / XDR",
]
DEFAULT_N_STRAIN = "DS / MDR"


PICKLE_UNCERTAINTIES = [
    "No saving or loading",
    "Load",
    "Save"
]
DEFAULT_PICKLE_UNCERTAINTY = "No saving or loading"


app = Flask(__name__)


class Form(FlaskForm):
    # Model running fields
    submit = SubmitField("Run")
    country = SelectField("Country", choices=zip(COUNTRIES, COUNTRIES), default=DEFAULT_COUNTRY)
    integration_method = SelectField("Integration method", choices=zip(INTEGRATION_METHODS, INTEGRATION_METHODS), default=DEFAULT_INTEGRATION_METHOD)
    fitting_method = SelectField("Fitting method", choices=zip(FITTING_METHODS, FITTING_METHODS), default=DEFAULT_FITTING_METHOD)
    fitting_smoothness = DecimalRangeField("Default fitting smoothness", places=1, default=0)
    integration_time_step = DecimalRangeField("Integration time step", places=3, default=0.005)

    # Model stratifications
    riskgroup_diabetes = BooleanField("Type II diabetes")
    riskgroup_hiv = BooleanField("HIV")
    riskgroup_prison = BooleanField("Prison", default="selected")
    riskgroup_indigenous = BooleanField('Indigenous')
    riskgroup_urbanpoor = BooleanField("Urban poor", default="selected")
    riskgroup_ruralpoor = BooleanField("Rural poor", default="selected")
    n_organs = SelectField("Number of organ strata", choices=zip(N_ORGANS, N_ORGANS), default=DEFAULT_N_ORGAN)
    n_strains = SelectField("Number of strains", choices=zip(N_STRAINS, N_STRAINS), default=DEFAULT_N_STRAIN)

    # Elaborations
    is_lowquality = BooleanField("Low quality care", default="selected")
    is_amplification = BooleanField("Resistance amplification", default="selected")
    is_misassignment = BooleanField("Strain mis-assignment", default="selected")
    is_vary_detection_by_organ = BooleanField("Vary case detection by organ status", default="selected")

    # Scenarios to run
    scenario_1 = BooleanField("Scenario 1")
    scenario_2 = BooleanField("Scenario 2")
    scenario_3 = BooleanField("Scenario 3")
    scenario_4 = BooleanField("Scenario 4")
    scenario_5 = BooleanField("Scenario 5")
    scenario_6 = BooleanField("Scenario 6")
    scenario_7 = BooleanField("Scenario 7")
    scenario_8 = BooleanField("Scenario 8")
    scenario_9 = BooleanField("Scenario 9")
    scenario_10 = BooleanField("Scenario 10")
    scenario_11 = BooleanField("Scenario 11")
    scenario_12 = BooleanField("Scenario 12")
    scenario_13 = BooleanField("Scenario 13")
    scenario_14 = BooleanField("Scenario 14")

    # Uncertainty
    output_uncertainty = BooleanField("Run uncertainty")
    adaptive_uncertainty = BooleanField("Adaptive search", default="selected")
    uncertainty_runs = IntegerField("Number of uncertainty runs", default=10)
    burn_in_runs = IntegerField("Number of burn-in runs", default=0)
    search_width = DecimalField("Relative search width", places=2, default=.08)
    pickle_uncertainty = SelectField("Pickle uncertainty", choices=zip(PICKLE_UNCERTAINTIES, PICKLE_UNCERTAINTIES), default=DEFAULT_PICKLE_UNCERTAINTY)

    # Plotting
    output_flow_diagram = BooleanField("Draw flow diagram")
    output_compartment_populations = BooleanField("Plot compartment sizes")
    output_riskgroup_fractions = BooleanField("Plot proportions by risk group")
    output_age_fractions = BooleanField("Plot proportions by age")
    output_by_subgroups = BooleanField("Plot outcomes by sub-groups")
    output_fractions = BooleanField("Plot compartment fractions")
    output_scaleups = BooleanField("Plot scale-up functions")
    output_gtb_plots = BooleanField("Plot outcomes")
    output_plot_economics = BooleanField("Plot economics graphs")
    output_plot_riskgroup_checks = BooleanField("Plot risk group checks")
    output_param_plots = BooleanField("Plot parameter progression")
    output_popsize_plot = BooleanField("Plot \"popsizes\" for cost-coverage curves")
    output_likelihood_plot = BooleanField("Plot log likelihoods over runs")
    output_age_calculations = BooleanField("Plot age calculation weightings")

    # MS Office outputs
    output_spreadsheets = BooleanField("Write to spreadsheets")
    output_documents = BooleanField("Write to documents")
    output_by_scenario = BooleanField("Output by scenario")
    output_horizontally = BooleanField("Write horizontally")

    class Meta:
        csrf = False


@app.route("/", methods=["GET", "POST"])
def index():
    form = Form()
    if form.validate_on_submit():
        return redirect(url_for("run"))
    return render_template("index.html", form=form)


@app.route("/run", methods=["GET", "POST"])
def run():
    return render_template("run.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

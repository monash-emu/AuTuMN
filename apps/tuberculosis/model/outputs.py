"""
FIXME: These all need tests.
"""
from typing import List, Dict
import numpy as np
from autumn.constants import Compartment


def calculate_percentage_susceptible(time_idx, model, compartment_values, derived_outputs):
    """
    Calculate the total number of susceptible people at each time-step.
    """
    prevalence_susceptible = 0
    for i, comp in enumerate(model.compartment_names):
        is_susceiptible = comp.has_name(Compartment.SUSCEPTIBLE)
        if is_susceiptible:
            prevalence_susceptible += compartment_values[i]
    population_size = sum(compartment_values)
    return 100. * prevalence_susceptible / population_size


def calculate_percentage_latent(time_idx, model, compartment_values, derived_outputs):
    """
    Calculate the percentage of individuals living with latent TB infection.
    """
    prevalence_latent = 0
    for i, comp in enumerate(model.compartment_names):
        is_latent = comp.has_name(Compartment.EARLY_LATENT) or comp.has_name(Compartment.LATE_LATENT)
        if is_latent:
            prevalence_latent += compartment_values[i]
    population_size = sum(compartment_values)
    return 100. * prevalence_latent / population_size


def calculate_prevalence_infectious(time_idx, model, compartment_values, derived_outputs):
    """
    Calculate active TB prevalence at each time-step. Returns the proportion per 100,000 population.
    """
    prevalence_infectious = 0
    for i, comp in enumerate(model.compartment_names):
        is_infectious = comp.has_name(Compartment.INFECTIOUS)
        if is_infectious:
            prevalence_infectious += compartment_values[i]
    population_size = sum(compartment_values)
    return 1.e5 * prevalence_infectious / population_size


def calculate_population_size(time_idx, model, compartment_values, derived_outputs):
    return sum(compartment_values)

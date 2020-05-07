"""
Constants used in building the AuTuMN / SUMMER models.
"""
import os

# Import summer constants here for convenience
from summer.constants import BirthApproach, IntegrationType, Stratification, Compartment, Flow

# Filesystem paths
file_path = os.path.abspath(__file__)
separator = "\\" if "\\" in file_path else "/"
BASE_PATH = separator.join(file_path.split(separator)[:-2])
DATA_PATH = os.path.join(BASE_PATH, "data")
EXCEL_PATH = os.path.join(DATA_PATH, "xls")

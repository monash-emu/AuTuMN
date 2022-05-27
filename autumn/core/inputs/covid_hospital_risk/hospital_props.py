import os
from autumn.settings.folders import INPUT_DATA_PATH
from autumn.core.project.params import read_yaml_file

def read_hospital_props(reference_strain: str) -> dict:
    """
    Load hospital proportion parameters for a given strain.
    """
    data_path = os.path.join(INPUT_DATA_PATH, "covid_hospital_risk", "covid_hospital_risk.yml")
    hospital_props = read_yaml_file(data_path)

    return hospital_props[reference_strain]
import os
import numpy as np
import pandas as pd


def load_specific_prem_sheet(file_type, sheet_name):
    """
    Load a mixing matrix sheet, according to user-specified name of the sheet (i.e. country)
    """
    file_dir = \
        os.path.join(os.path.abspath(
            os.getcwd()),
            'social_mixing_data',
            'MUestimates_' + file_type + '.xlsx'
        )
    return np.array(pd.read_excel(file_dir, sheet_name=sheet_name))


def load_all_prem_sheets(file_type):
    """
    Load all the mixing matrices (i.e. sheets) from a specified excel file
    """
    file_dir = \
        os.path.join(os.path.abspath(
            os.getcwd()),
            'social_mixing_data',
            'MUestimates_' + file_type + '.xlsx'
        )
    excel_file = pd.ExcelFile(file_dir)
    matrices = {}
    for sheet in excel_file.sheet_names:
        matrices[sheet] = load_specific_prem_sheet(file_type, sheet)
    return matrices

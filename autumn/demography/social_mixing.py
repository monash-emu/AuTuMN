import os
import numpy as np
import pandas as pd


def load_specific_prem_sheet(mixing_location, country):
    """
    Load a mixing matrix sheet, according to name of the sheet (i.e. country)

    :param: mixing_location: str
        One of the four mixing locations - ('all_locations', 'home', 'other_locations', 'school', 'work')
    :param: country: str
        Name of the country of interest
    """

    # Files with name ending with _1 have a header, but not those ending with _2 - plus need to determine file to read
    sheet_number, header_argument = ('1', 0) if country < 'Mozambique' else ('2', None)

    file_dir = \
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'social_mixing_data',
            'MUestimates_' + mixing_location + '_' + sheet_number + '.xlsx'
        )

    return np.array(pd.read_excel(file_dir, sheet_name=country, header=header_argument))


def load_all_prem_types(country):
    """
    Collate the matrices of different location types for a given country

    :param country: str
        Name of the requested country
    """
    matrices = {}
    for sheet_type in ('all_locations', 'home', 'other_locations', 'school', 'work'):
        matrices[sheet_type] = load_specific_prem_sheet(sheet_type, country)
    return matrices


def load_age_calibration():
    """
    converts the age group specific cases to covid_19 agegroup
    0–9,    10–19,  20–29,  30–39,  40–49,  50–59,  60–69,  70–79,  80+
    2,      2,	    13,	    11,	    11,	    14,	    8,	    6,	    4

    Returns:
        a pandas series
    """

    age_breakpoints = [int(i_break) for i_break in list(range(0, 80, 5))]
        
    # split the case numbers into 5 year groups
    case_numbers = [2, 2, 13, 11, 11, 14, 8, 6, 4]
    case_numbers = [each/2 for each in case_numbers for y in range(2)]

    # create case numbers for 75+
    y = case_numbers[:-3]
    y.append(sum(case_numbers[-3:]))
  
    return pd.Series(y, index=age_breakpoints)


def change_mixing_matrix_for_scenario(model, mixing_functions, i_scenario):
    """
    Change the mixing matrix to a dynamic version to reflect interventions acting on the mixing matrix
    """
    if mixing_functions and i_scenario in mixing_functions:
        model.find_dynamic_mixing_matrix = mixing_functions[i_scenario]
        model.dynamic_mixing_matrix = True
    return model


def update_mixing_with_multipliers(mixing_matrix, multipliers):
    """
    Updates the mixing matrix using some age-specific multipliers
    :param mixing_matrix: the baseline mixing-matrix
    :param multipliers: a matrix with the ages-specific multipliers
    :return: the updated mixing-matrix
    """
    assert mixing_matrix.shape == multipliers.shape

    return np.multiply(mixing_matrix, multipliers)


def get_all_prem_countries():
    """
    Return the list of countries for which Prem et al provide contact matrices
    """
    sheet_names = []
    for file_number in ('1', '2'):
        filepath = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                'social_mixing_data',
                'MUestimates_all_locations_' + file_number + '.xlsx'
            )
        xl = pd.ExcelFile(filepath)
        sheet_names += xl.sheet_names
    return sheet_names


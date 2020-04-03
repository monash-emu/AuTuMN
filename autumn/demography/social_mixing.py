import os
import numpy as np
import pandas as pd


def load_specific_prem_sheet(file_type, sheet_name):
    """
    Load a mixing matrix sheet, according to user-specified name of the sheet (i.e. country)
    """
    file_dir = \
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'social_mixing_data',
            'MUestimates_' + file_type + '.xlsx'
        )
    return np.array(pd.read_excel(file_dir, sheet_name=sheet_name))


def load_all_prem_types(country, sheet_group):
    """
    Collate the matrices of different location types for a given country

    :param country: str
        Name of the requested country
    :param sheet_group: int
        Either 1 or 2 depending on the country's position in the alphabet
    """
    matrices = {}
    for sheet_type in ['all_locations', 'home', 'other_locations', 'school', 'work']:
        matrices[sheet_type] = load_specific_prem_sheet(sheet_type + '_' + str(sheet_group), country)
    return matrices


def load_all_prem_sheets(file_type):
    """
    Load all the mixing matrices (i.e. sheets) from a specified excel file
    """
    file_dir = \
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'social_mixing_data',
            'MUestimates_' + file_type + '.xlsx'
        )
    excel_file = pd.ExcelFile(file_dir)
    matrices = {}
    for sheet in excel_file.sheet_names:
        matrices[sheet] = load_specific_prem_sheet(file_type, sheet)
    return matrices


def load_population(file, sheet_name):
    """
    Load the population of Australia as provided by ABS at June 2019

    Args:
        file: excel file name with extension e.g pop.xls
        sheet_name: sheet name within the file

    Returns:
        Pandas data frame with population numbers by 5 year age groups, gender and state.
    """

    file_dir = \
        os.path.join(os.getcwd(), 'applications\covid_19\covid specific data',file)

    population = pd.read_excel(file_dir,sheet_name,4)

    # find the row numbers to break the DF along
    male_end = population.index[population['Age group (years)'] == 'FEMALES'].tolist()
    female_end = population.index[population['Age group (years)'] == 'PERSONS'].tolist()

    m_pop = population.iloc[:male_end[0],:]
    f_pop = population.iloc[male_end[0]:female_end[0],:]
    p_pop = population.iloc[female_end[0]:,:]

    m_pop.loc[:,'Gender'] = 'Male'
    f_pop.loc[:,'Gender'] = 'Female'
    p_pop.loc[:,'Gender'] = 'Persons'

    # Remove top and last few rows of each DF
    m_pop.drop([0,22],inplace = True)
    f_pop.drop([23,45],inplace = True)
    p_pop.drop([46,68,69,70,71],inplace = True)

    # Make one DF and create a multi-level index
    population = pd.concat([m_pop,f_pop,p_pop])
    population.set_index(['Age group (years)','Gender'], inplace = True)

    over_75 =  population.loc[[('75–79'),('80–84'),('85–89'),('85–89'),('90–94'),('95–99'),('100 and over'),],:]
    population.drop([('75–79'),('80–84'),('85–89'),('85–89'),('90–94'),('95–99'),('100 and over')], inplace =True)

    over_75 =  over_75.groupby('Gender').sum()
    over_75['Age group (years)'] = '75+'
    over_75.reset_index(inplace =True)
    over_75.set_index(['Age group (years)','Gender'], inplace =True)

    population = pd.concat([population,over_75])
    
    
    return population


def load_age_calibration():
    '''
    converts the age group specific cases to covid_19 agegroup
    0–9,    10–19,  20–29,  30–39,  40–49,  50–59,  60–69,  70–79,  80+
    2,      2,	    13,	    11,	    11,	    14,	    8,	    6,	    4

    Returns:
        a pandas series
    '''

    age_breakpoints = [int(i_break) for i_break in list(range(0, 80, 5))]
        
    # split the case numbers into 5 year groups
    case_numbers = [2,2,13,11,11,14,8,6,4]
    case_numbers = [ each/2 for each in case_numbers for y in range(2) ]

    # create case numbers for 75+
    y = case_numbers[:-3]
    y.append(sum(case_numbers[-3:]))
  
    return pd.Series(y, index=age_breakpoints)


def change_mixing_matrix_for_scenario(model, scenario_requests, default_params):
    """
    Change the mixing matrix to a dynamic version to reflect interventions
    """
    mixing_matrix_components = load_all_prem_types(default_params['country'], 1)

    def mixing_matrix_function(time):
        mixing_matrix = mixing_matrix_components['all_locations']
        for location in \
                [loc for loc in ['home', 'other_locations', 'school', 'work'] if loc in scenario_requests['mixing']]:
            school_closure_change = \
                np.piecewise(
                    time,
                    [time < default_params['school_closure'],
                     default_params['school_closure'] <= time < default_params['school_reopening'],
                     default_params['school_reopening'] <= time],
                    [0., scenario_requests['mixing'][location], 0.]
                )
            mixing_matrix = np.add(mixing_matrix, school_closure_change * mixing_matrix_components[location])
        return mixing_matrix

    model.find_dynamic_mixing_matrix = mixing_matrix_function
    model.dynamic_mixing_matrix = True
    return model


def get_all_prem_countries():
    """
    Return the list of countries for which Prem et al provide contact matrices
    """
    sheet_names = []
    for file_number in ['1', '2']:
        filepath = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                'social_mixing_data',
                'MUestimates_all_locations_' + file_number + '.xlsx'
            )
        xl = pd.ExcelFile(filepath)
        sheet_names += xl.sheet_names
    return sheet_names

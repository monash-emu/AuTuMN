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
    
    return population
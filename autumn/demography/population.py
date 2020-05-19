import os
import pandas as pd


def load_population(file, sheet_name):
    """
    Load the population of Australia as provided by ABS at June 2019

    Args:
        file: excel file name with extension e.g pop.xls
        sheet_name: sheet name within the file

    Returns:
        Pandas data frame with population numbers by 5 year age groups, gender and state.
    """

    file_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), file)

    population = pd.read_excel(file_dir, sheet_name, 4)

    # find the row numbers to break the DF along
    male_end = population.index[population["Age group (years)"] == "FEMALES"].tolist()
    female_end = population.index[population["Age group (years)"] == "PERSONS"].tolist()

    m_pop = population.iloc[: male_end[0], :]
    f_pop = population.iloc[male_end[0] : female_end[0], :]
    p_pop = population.iloc[female_end[0] :, :]

    m_pop.loc[:, "Gender"] = "Male"
    f_pop.loc[:, "Gender"] = "Female"
    p_pop.loc[:, "Gender"] = "Persons"

    # Remove top and last few rows of each DF
    m_pop.drop([0, 22], inplace=True)
    f_pop.drop([23, 45], inplace=True)
    p_pop.drop([46, 68, 69, 70, 71], inplace=True)

    # Make one DF and create a multi-level index
    population = pd.concat([m_pop, f_pop, p_pop])
    population.set_index(["Age group (years)", "Gender"], inplace=True)

    over_75 = population.loc[
        [("75–79"), ("80–84"), ("85–89"), ("85–89"), ("90–94"), ("95–99"), ("100 and over"),], :
    ]
    population.drop(
        [("75–79"), ("80–84"), ("85–89"), ("85–89"), ("90–94"), ("95–99"), ("100 and over")],
        inplace=True,
    )

    over_75 = over_75.groupby("Gender").sum()
    over_75["Age group (years)"] = "75+"
    over_75.reset_index(inplace=True)
    over_75.set_index(["Age group (years)", "Gender"], inplace=True)
    population = pd.concat([population, over_75])
    return population

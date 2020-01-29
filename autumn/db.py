
from sqlalchemy import create_engine
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from autumn_from_summer.curve import scale_up_function


def get_bcg_coverage(database, country_iso_code):
    """
    extract bcg coverage from inputs database

    :param database: sql database
        database containing the bcg data
    :param country_iso_code: string
        three letter ISO3 code for the country of interest
    :return: dict
        pandas data frame with columns years and one row containing the values of BCG coverage in that year
    """
    _bcg_coverage = database.db_query("BCG", conditions=["ISO_code='" + country_iso_code + "'"])
    _bcg_coverage = _bcg_coverage.filter(items=[column for column in _bcg_coverage.columns if column.isdigit()])
    return {int(key): value / 1e2 for key, value in zip(list(_bcg_coverage.columns), _bcg_coverage.loc[0, :])
            if value is not None}


def get_all_iso3_from_bcg(database):
    """
    check which iso3 country codes are available from the bcg database

    :param database: sql database
        the database containing the bcg data
    :return: list
        all the iso3 strings available from the bcg database
    """

    return database.db_query("bcg", column="ISO_code")["ISO_code"].tolist()


def get_crude_birth_rate(database, country_iso_code):
    """
    get the crude birth rate as a rate estimated by the un for a particular country

    :param database: sql database
        database containing the crude birth rate data from un
    :param country_iso_code: string
        three letter ISO3 code for the country of interest
    :return: dict
        keys for mid-point of year of interest with floats for the crude birth rate (per capita, not per 1,000)
    """

    # extract birth rates
    birth_rates = database.db_query("crude_birth_rate_mapped", conditions=["iso3='" + country_iso_code +"'"])

    # find the keys with a - in them to indicate a time range and add 2.5 on to the starting value to get mid-point
    return {float(key[: key.find("-")]) + 2.5: float(value) / 1e3 for
            key, value in zip(list(birth_rates.columns), birth_rates.loc[0, :]) if "-" in key}


def extract_demo_data(_input_database, data_type, country_iso_code):
    """
    get and format demographic data from the input databases originally derived from the un sources
    note that the number of period that data are provided for differs for total population and absolute deaths

    :param _input_database: sql database
        the master inputs database
    :param data_type: str
        the database type of interest
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :return: pandas dataframe
        cleaned pandas dataframe ready for use in demographic calculations
    """

    # get the appropriate data type from the un-derived databases
    demo_data_frame = _input_database.db_query(data_type, conditions=["iso3='" + country_iso_code +"'"])

    # rename columns, including adding a hyphen to the last age group to make it behave like the others age groups
    demo_data_frame.rename(columns={"95+": "95-", "Reference date (as of 1 July)": "Period"}, inplace=True)

    # retain only the relevant columns
    columns_to_keep = [column for column in demo_data_frame.columns if "-" in column]
    columns_to_keep.append("Period")
    demo_data_frame = demo_data_frame.loc[:, columns_to_keep]

    # rename the columns to make them integers
    demo_data_frame.columns = \
        [int(column[:column.find("-")]) if "-" in column else column for column in list(demo_data_frame.columns)]

    # change the year data for the period to numeric type
    demo_data_frame["Period"] = \
        demo_data_frame["Period"].apply(lambda x: str(x)[: str(x).find("-")] if "-" in str(x) else str(x))

    # return final version
    return demo_data_frame


def prepare_age_breakpoints(breakpoints):
    """
    temporary function - should merge in with functions from the summer module

    :param breakpoints:
    :return:
    """
    breakpoints.sort()
    return breakpoints if 0 in breakpoints else [0] + breakpoints


def find_death_rates(_input_database, country_iso_code):
    """
    find death rates by reported age bracket from database populated from un data

    :param _input_database: sql database
        the inputs database
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :return: pandas dataframe:
        mortality rates by age bracket
        mortality_years: list
        values of the mid-points of the years for which mortality is estimated
    """

    # get necessary data from database
    absolute_death_data = extract_demo_data(_input_database, "absolute_deaths_mapped", country_iso_code)
    total_population_data = extract_demo_data(_input_database, "total_population_mapped", country_iso_code)

    # cut off last row of population data because it goes out five years longer
    total_population_data = total_population_data.loc[:absolute_death_data.shape[0] - 1, :]

    # cut off last column of both data frames because they include the years, but retain the data as a list
    mortality_years = [float(i) + 2.5 for i in list(total_population_data.loc[:, "Period"])]
    total_population_data = total_population_data.iloc[:, :total_population_data.shape[1] - 1]
    absolute_death_data = absolute_death_data.iloc[:, :absolute_death_data.shape[1] - 1]

    # make sure all floats, as seem to have become str somewhere
    absolute_death_data = absolute_death_data.astype(float)
    total_population_data = total_population_data.astype(float)

    # replace NaN and inf values with zeros
    death_rates = absolute_death_data / total_population_data / 5.0
    for col in death_rates.columns:
        death_rates[col] = death_rates[col].fillna(0)
        death_rates[col] = death_rates[col].replace(np.inf, 0.)

    # divide through by population and by five to allow for the mortality data being aggregated over five year periods
    return death_rates, mortality_years


def find_age_weights(age_breakpoints, demo_data, arbitrary_upper_age=1e2, break_width=5.0):
    """
    find the weightings to assign to the various components of the data from the age breakpoints planned to be used
    in the model

    :param age_breakpoints: list
        numeric values for the breakpoints of the age brackets
    :param demo_data: pandas dataframe
        the demographic data extracted from the database into pandas format
    :param arbitrary_upper_age: float
        arbitrary upper value to consider for the highest age bracket
    :param break_width: float
        difference between the lower and upper values of the age brackets in the data
    :return: dict
        keys age breakpoints, values list of the weightings to assign to the data age categories
    """
    weightings_dict = {}

    # cycle through each age bracket/category
    for n_breakpoint in range(len(age_breakpoints)):

        lower_value = age_breakpoints[n_breakpoint]
        upper_value = arbitrary_upper_age if n_breakpoint == len(age_breakpoints) - 1 else \
            age_breakpoints[n_breakpoint + 1]

        # initialise weights to one and then subtract parts of bracket that are excluded
        weightings = [1.0] * len(demo_data.columns)

        # cycle through the breakpoints of the data on inner loop
        for n_data_break, data_breakpoints in enumerate(demo_data.columns):
            data_lower = data_breakpoints
            data_upper = data_breakpoints + break_width

            # first consider the lower value of the age bracket and how much of the data it excludes
            if data_upper <= lower_value:
                weightings[n_data_break] -= 1.0
            elif data_lower < lower_value < data_upper:
                weightings[n_data_break] -= 1.0 - (data_upper - lower_value) / break_width

            # then consider the upper value of the age bracket and how much of the data it excludes
            if data_lower < upper_value < data_upper:
                weightings[n_data_break] -= 1.0 - (upper_value - data_lower) / break_width
            elif upper_value <= data_lower:
                weightings[n_data_break] -= 1.0

        # normalise the values
        weightings = [weight / sum(weightings) for weight in weightings]
        weightings_dict[age_breakpoints[n_breakpoint]] = weightings
    return weightings_dict


def find_age_specific_death_rates(input_database, age_breakpoints, country_iso_code):
    """
    find non-tb-related death rates from un data that are specific to the age groups requested for the model regardless
    of the age brackets for which data are available

    :param age_breakpoints: list
        integers for the age breakpoints being used in the model
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :return: dict
        keys the age breakpoints, values lists for the death rates with time
    """
    age_breakpoints = prepare_age_breakpoints(age_breakpoints)

    # gather up the death rates with the brackets from the data
    death_rates, years = find_death_rates(input_database, country_iso_code)

    # find the weightings to each age group in the data from the requested brackets
    age_weights = find_age_weights(age_breakpoints, death_rates)

    # calculate the list of values for the weighted death rates for each modelled age category
    age_death_rates = {}
    for age_break in age_breakpoints:
        age_death_rates[age_break] = [0.0] * death_rates.shape[0]
        for year in death_rates.index:
            age_death_rates[age_break][year] = \
                sum([death_rate * weight for death_rate, weight in
                     zip(list(death_rates.iloc[year]), age_weights[age_break])])
    return age_death_rates, years


def get_pop_mortality_functions(input_database, age_breaks, country_iso_code, emigration_value=0.0,
                                emigration_start_time=1980.):
    """
    use the mortality rate data that can be obtained from find_age_specific_death_rates to fit time-variant mortality
        functions for each age group being implemented in the model

    :param age_breaks: list
        starting ages for each of the age groups
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :param emigration_value: float
        an extra rate of migration to add on to the population-wide mortality rates to simulate net emigration
    :param emigration_start_time: float
        the point from which the additional net emigration commences
    :return: dict
        keys age breakpoints, values mortality functions
    """
    age_death_dict, data_years = find_age_specific_death_rates(input_database, age_breaks, country_iso_code)

    # add an extra fixed value after a particular time point for each mortality estimate
    for age_group in age_death_dict:
        for i_year in range(len(age_death_dict[age_group])):
            if data_years[i_year] > emigration_start_time:
                age_death_dict[age_group][i_year] += emigration_value

    # fit the curve functions to the aggregate data of mortality and net emigration
    return {age_group: scale_up_function(data_years, age_death_dict[age_group], smoothness=0.2, method=5) for
            age_group in age_death_dict}


class InputDB:
    """
    methods for loading input xls files
    """

    def __init__(self, database_name="databases/inputs.db", verbose=False):
        """
        initialise sqlite database
        """
        self.database_name = database_name
        self.engine = create_engine("sqlite:///" + database_name, echo=False)
        self.verbose = verbose
        self.headers_lookup = \
            {"xls/WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx": 16,
             "xls/WPP2019_F01_LOCATIONS.xlsx": 16,
             "xls/WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx": 16,
             "xls/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx": 16,
             "xls/life_expectancy_2015.xlsx": 3,
             "xls/rate_birth_2015.xlsx": 3}
        self.tab_of_interest = \
            {"xls/WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx": "ESTIMATES",
             "xls/WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx": "ESTIMATES",
             "xls/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx": "ESTIMATES",
             "xls/WPP2019_F01_LOCATIONS.xlsx": "Location",
             "xls/coverage_estimates_series.xlsx": "BCG",
             "xls/gtb_2015.xlsx": "gtb_2015",
             "xls/gtb_2016.xlsx": "gtb_2016",
             "xls/life_expectancy_2015.xlsx": "life_expectancy_2015",
             "xls/rate_birth_2015.xlsx": "rate_birth_2015"}
        self.output_name = \
            {"xls/WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx": "crude_birth_rate",
             "xls/WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx": "absolute_deaths",
             "xls/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx": "total_population",
             "xls/WPP2019_F01_LOCATIONS.xlsx": "un_iso3_map",
             "xls/coverage_estimates_series.xlsx": "bcg",
             "xls/gtb_2015.xlsx": "gtb_2015",
             "xls/gtb_2016.xlsx": "gtb_2016",
             "xls/life_expectancy_2015.xlsx": "life_expectancy_2015",
             "xls/rate_birth_2015.xlsx": "rate_birth_2015"}
        self.map_df = None

    def update_csv_reads(self, input_path="xls/*.csv"):
        """
        load csvs from input_path
        """
        csv_file_list = glob.glob(input_path)
        for filename in csv_file_list:
            data_frame = pd.read_csv(filename)
            data_frame.to_sql(filename.split("\\")[1].split(".")[0], con=self.engine, if_exists="replace")

    def update_xl_reads(self, sheets_to_read=glob.glob("xls/*.xlsx")):
        """
        load excel spreadsheet from input_path

        :param sheets_to_read: iterable
            paths of the spreadsheets to read, which have to be strictly coded in the format suggested above
        """
        for available_file in sheets_to_read:
            filename = "xls/" + available_file[4: -5] + ".xlsx"
            header_row = self.headers_lookup[filename] if filename in self.headers_lookup else 0
            data_title = self.output_name[filename] if filename in self.output_name else filename
            current_data_frame = pd.read_excel(
                pd.ExcelFile(filename), header=header_row, index_col=1, sheet_name=self.tab_of_interest[filename])
            self.output_to_user("now reading '%s' tab of '%s' file" % (self.tab_of_interest[filename], filename))
            current_data_frame.to_sql(data_title, con=self.engine, if_exists="replace")

    def output_to_user(self, comment):
        """
        report progress to user if requested

        :param comment: str
            string to be output to the user
        """
        if self.verbose:
            print(comment)

    def db_query(self, table_name, column="*", conditions=[]):
        """
        method to query table_name

        :param table_name: str
            name of the database table to query from
        :param conditions: str
            list of SQL query conditions (e.g. ["Scenario='1'", "idx='run_0'"])
        :param value: str
            value of interest with filter column
        :param column:

        :return: pandas dataframe
            output for user
        """
        query = "SELECT %s FROM %s" % (column, table_name)
        if len(conditions) > 0:
            query += " WHERE"
            for condition in conditions:
                query += ' ' + condition
        query += ";"
        return pd.read_sql_query(query, con=self.engine)

    def add_iso_to_table(self, table_name):
        """
        add the mapped iso3 code to a table that only contains the un country code

        :param table_name: str
            name of the spreadsheet to perform this on
        """

        # perform merge
        self.get_un_iso_map()
        table_with_iso = pd.merge(
            self.db_query(table_name=table_name), self.map_df, left_on='Country code', right_on='Location code')

        # columns with spaces are difficult to read with sql queries
        table_with_iso.rename(columns={"ISO3 Alpha-code": "iso3"}, inplace=True)

        # remove index column to avoid creating duplicates
        if "Index" in table_with_iso.columns:
            table_with_iso = table_with_iso.drop(columns=["Index"])

        # create new mapped database structure
        table_with_iso.to_sql(table_name + "_mapped", con=self.engine, if_exists="replace")

    def get_un_iso_map(self):
        """
        create dictionary structure to map from un three numeric digit codes to iso3 three alphabetical digit codes
        """
        self.map_df = self.db_query(table_name='un_iso3_map')[['Location code', 'ISO3 Alpha-code']].dropna()


if __name__ == "__main__":

    # standard code to update the database
    input_database = InputDB(database_name='databases/Inputs.db')
    #input_database.update_xl_reads()
    # input_database.add_iso_to_table("crude_birth_rate")
    # input_database.add_iso_to_table("absolute_deaths")
    # input_database.add_iso_to_table("total_population")
    # input_database.update_csv_reads()

    # example_age_breakpoints = [10, 3]
    # pop_morts = get_pop_mortality_functions(input_database, example_age_breakpoints, country_iso_code="MNG")

    # example of accessing once loaded
    # times = list(np.linspace(1950, 2020, 1e3))
    # extract data for BCG vaccination for a particular country
    # for country in get_all_iso3_from_bcg(input_database):
    #     bcg_coverage = get_bcg_coverage(input_database, country)
    #     if len(bcg_coverage) == 0:
    #         print("no BCG vaccination data available for %s" % country)
    #         continue
    #     print("plotting BCG vaccination data and fitted curve for %s" % country)
    #     bcg_coverage_function = scale_up_function(
    #           bcg_coverage.keys(), bcg_coverage.values(), smoothness=0.2, method=5)
    #     plt.plot(list(bcg_coverage.keys()), list(bcg_coverage.values()), "ro")
    #     plt.plot(times, [bcg_coverage_function(time) for time in times])
    #     plt.title(country)
    #     plt.show()

    # times = list(np.linspace(1950, 2020, 1e3))
    # crude_birth_rate_data = get_crude_birth_rate(input_database, "MNG")
    # birth_rate_function = \
    #     scale_up_function(crude_birth_rate_data.keys(), crude_birth_rate_data.values(), smoothness=0.2, method=5)
    # plt.plot(list(crude_birth_rate_data.keys()), list(crude_birth_rate_data.values()), "ro")
    # plt.plot(times, [birth_rate_function(time) for time in times])
    # plt.title("MNG")
    # plt.show()

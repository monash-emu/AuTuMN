import glob

import pandas as pd
from sqlalchemy import create_engine


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
    # FIXME: Not sure what to do with this.
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

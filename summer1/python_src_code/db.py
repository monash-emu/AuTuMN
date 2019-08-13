
from sqlalchemy import create_engine
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from python_source_code.curve import scale_up_function


def get_bcg_coverage(country_iso_code):
    """
    extract bcg coverage from inputs database

    :param country_iso_code: string
        three letter ISO3 code for the country of interest
    :return: bcg_coverage
        pandas data frame with columns years and one row containing the values of BCG coverage in that year
    """
    _bcg_coverage = input_database.db_query("BCG", is_filter="ISO_code", value=country_iso_code)
    _bcg_coverage = _bcg_coverage.filter(items=[column for column in _bcg_coverage.columns if column.isdigit()])
    return {int(key): value for key, value in zip(list(_bcg_coverage.columns), _bcg_coverage.loc[0, :])
            if value is not None}


def get_all_iso3_from_bcg(database):
    return database.db_query("bcg", column="ISO_code")["ISO_code"].tolist()


class InputDB:
    """
    methods for loading input xls files
    """

    def __init__(self, database_name="../databases/inputs.db", verbose=False):
        """
        initialise sqlite database
        """
        self.database_name = database_name
        self.engine = create_engine("sqlite:///" + database_name, echo=False)
        self.verbose = verbose
        self.tabs_of_interest = ["BCG", "Aggregated estimates"]

    def update_csv_reads(self, input_path="../xls/*.csv"):
        """
        load csvs from input_path
        """
        csv_file_list = glob.glob(input_path)
        for filename in csv_file_list:
            data_frame = pd.read_csv(filename)
            data_frame.to_sql(filename.split("\\")[1].split(".")[0], con=self.engine, if_exists="replace")

    def update_xl_reads(self, input_path="../xls/*.xlsx"):
        """
        load excel spreadsheet from input_path
        """
        excel_file_list = glob.glob(input_path)
        for filename in excel_file_list:
            xls = pd.ExcelFile(filename)

            # for single tab in spreadsheet
            if len(xls.sheet_names) == 1:
                df_name = xls.sheet_names[0]
                df = pd.read_excel(filename, sheet_name=df_name)
                df.to_sql(df_name, con=self.engine, if_exists="replace")
                self.output_to_user("now reading '%s' tab of '%s' file" % (df_name, filename))

            # if multiple tabs
            else:
                for n_sheets, sheet in enumerate(xls.sheet_names):
                    if sheet in self.tabs_of_interest:
                        header_3_sheets = ["rate_birth_2015", "life_expectancy_2015"]
                        n_header = 3 if sheet in header_3_sheets else 0
                        df = pd.read_excel(filename, sheet_name=sheet, header=n_header)
                        self.output_to_user("now reading '%s' tab of '%s' file" % (sheet, filename))

                        # to read constants and time variants
                        if sheet == "constants":
                            sheet = filename.replace(".xlsx", "").split("_")[1] + "_constants"
                        if sheet == "time_variants":
                            sheet = filename.replace(".xlsx", "").split("_")[1] + "_time_variants"
                        df.to_sql(sheet, con=self.engine, if_exists="replace")

    def output_to_user(self, comment):
        """
        report progress to user if requested
        """
        if self.verbose:
            print(comment)

    def db_query(self, table_name, is_filter="", value="", column="*"):
        """
        method to query table_name
        """
        query = "Select %s from %s" % (column, table_name)
        if is_filter and value:
            query = query + " Where %s = \'%s\'" % (is_filter, value)
        return pd.read_sql_query(query, con=self.engine)


if __name__ == "__main__":

    # standard code to update the database
    input_database = InputDB(verbose=True)
    # input_database.update_xl_reads()
    # input_database.update_csv_reads()

    # example of accessing once loaded
    result = input_database.db_query("gtb_2015", column="c_cdr", is_filter="iso3", value="MNG")
    cdr_mongolia = result["c_cdr"].values
    result = input_database.db_query("gtb_2015", column="year", is_filter="iso3", value="MNG")
    cdr_mongolia_year = result["year"].values
    spl = scale_up_function(cdr_mongolia_year, cdr_mongolia, smoothness=0.2, method=5)
    times = list(np.linspace(1950, 2014, 1e3))
    scaled_up_cdr = [spl(t) for t in times]
    # plt.plot(cdr_mongolia_year, cdr_mongolia, "ro", times, scaled_up_cdr)
    # plt.title("CDR from GTB 2015")
    # plt.show()

    # extract data for BCG vaccination for a particular country
    # for country in get_all_iso3_from_bcg(input_database):
    #     bcg_coverage = get_bcg_coverage(country)
    #     if len(bcg_coverage) == 0:
    #         print("no BCG vaccination data available for %s" % country)
    #         continue
    #     print("plotting BCG vaccination data and fitted curve for %s" % country)
    #     bcg_coverage_function = scale_up_function(bcg_coverage.keys(), bcg_coverage.values(), smoothness=0.2, method=5)
    #     plt.plot(list(bcg_coverage.keys()), list(bcg_coverage.values()), "ro")
    #     plt.plot(times, [bcg_coverage_function(time) for time in times])
    #     plt.title(country)
    #     plt.show()

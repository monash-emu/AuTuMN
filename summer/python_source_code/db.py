
from sqlalchemy import create_engine
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from python_source_code.curve import scale_up_function


class InputDB:
    """
    methods for loading input xls files
    """

    def __init__(self, database_name="../databases/Inputs.db", report=False):
        """
        initialise sqlite database
        """
        self.database_name = database_name
        self.engine = create_engine("sqlite:///" + database_name, echo=False)
        self.report = report
        self.available_sheets \
            = ["default_constants", "country_constants", "default_programs", "country_programs", "bcg_2014", "bcg_2015",
               "bcg_2016", "rate_birth_2014", "rate_birth_2015", "life_expectancy_2014", "life_expectancy_2015",
               "notifications_2014", "notifications_2015", "notifications_2016", "outcomes_2013", "outcomes_2015",
               "mdr_2014", "mdr_2015", "mdr_2016", "laboratories_2014", "laboratories_2015", "laboratories_2016",
               "strategy_2014", "strategy_2015", "strategy_2016", "diabetes", "gtb_2015", "gtb_2016", "latent_2016",
               "tb_hiv_2016", "spending_inputs", "constants", "time_variants"]

    def load_csv(self, input_path="../xls/*.csv"):
        """
        load csvs from input_path
        """
        csv_file_list = glob.glob(input_path)
        for filename in csv_file_list:
            data_frame = pd.read_csv(filename)
            data_frame.to_sql(filename.split("\\")[1].split(".")[0], con=self.engine, if_exists="replace")

    def load_xlsx(self, input_path="../xls/*.xlsx"):
        """
        load excel spreadsheet from input_path
        """
        excel_file_list = glob.glob(input_path)
        for filename in excel_file_list:
            xls = pd.ExcelFile(filename)

            # for single work sheets
            if len(xls.sheet_names) == 1:
                df_name = xls.sheet_names[0]
                df = pd.read_excel(filename, sheet_name=df_name)
                df.to_sql(df_name, con=self.engine, if_exists="replace")
            else:
                n_sheets = 0
                while n_sheets < len(xls.sheet_names):
                    sheet_name = xls.sheet_names[n_sheets]
                    if sheet_name in self.available_sheets:
                        header_3_sheets = ["rate_birth_2015", "life_expectancy_2015"]
                        n_header = 3 if sheet_name in header_3_sheets else 0
                        df = pd.read_excel(filename, sheet_name=sheet_name, header=n_header)
                        self.output_to_user("now reading %s" % sheet_name)

                        # to read constants and time variants
                        if sheet_name == "constants":
                            sheet_name = filename.replace(".xlsx", "").split("_")[1] + "_constants"
                        if sheet_name == "time_variants":
                            sheet_name = filename.replace(".xlsx", "").split("_")[1] + "_time_variants"
                        df.to_sql(sheet_name, con=self.engine, if_exists="replace")
                    n_sheets += 1

    def output_to_user(self, comment):
        """
        report progress to user if requested
        """
        if self.report:
            print(comment)

    def db_query(self, table_name, is_filter="", value="", column="*"):
        """
        method to query table_name
        """
        query = "Select %s from  %s" % (column, table_name)
        if is_filter and value:
            query = query + " Where %s = \'%s\'" % (is_filter, value)
        return pd.read_sql_query(query, con=self.engine)


if __name__ == "__main__":

    input_database = InputDB(report=True)
    input_database.load_xlsx()
    input_database.load_csv()

    res = input_database.db_query("gtb_2015", column="c_cdr", is_filter="country", value="Mongolia")
    cdr_mongolia = res["c_cdr"].values
    res = input_database.db_query("gtb_2015", column="year", is_filter="country", value="Mongolia")
    cdr_mongolia_year = res["year"].values
    spl = scale_up_function(cdr_mongolia_year, cdr_mongolia, smoothness=0.2, method=5)
    times = list(np.linspace(1950, 2014, 1e3))
    scaled_up_cdr = []
    for t in times:
        scaled_up_cdr.append(spl(t))
    # print(scaled_up_cdr)
    plt.plot(cdr_mongolia_year, cdr_mongolia, "ro", times, scaled_up_cdr)
    plt.title("CDR from GTB 2015")
    plt.show()

    # res = input_database.db_query("bcg_2015", is_filter="Cname", value="Bhutan")
    # print(res)
    # res = input_database.db_query("notifications_2016", is_filter="Country", value="Bhutan")
    # print(res)
    # res = input_database.db_query("default_time_variants", is_filter="program", value="econ_cpi")
    # print(res)
    # res = input_database.db_query("bhutan_constants", is_filter="parameter", value="tb_n_contact")
    # print(res)
    # res = input_database.db_query("bhutan_time_variants", is_filter="program", value="int_perc_firstline_dst")
    # print(res.values)

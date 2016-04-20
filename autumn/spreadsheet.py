# -*- coding: utf-8 -*-

from __future__ import print_function
from xlrd import open_workbook # For opening Excel workbooks
from numpy import nan
import copy


"""
Import model inputs from Excel spreadsheet 
"""


def is_all_same_value(a_list, test_val):
    for val in a_list:
        if val != test_val:
            return False
    return True


def replace_blanks(a_list, new_val, blank):
    return [new_val if val == blank else val for val in a_list]


def parse_year_data(these_data, blank):
    these_data = replace_blanks(these_data, nan, blank)
    assumption_val = these_data[-1]
    year_vals = these_data[:-2] 
    if is_all_same_value(year_vals, nan):
        return [assumption_val] 
    else:
        # skip "OR" and assumption col
        return year_vals


class MacroeconomicsSheetReader:

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.name = 'macroeconomics'
        self.key = 'macro'
        self.parlist =  [
            'cpi',
            'ppp',
            'gdp',
            'govrevenue',
            'govexpen',
            'totdomesintlexpen',
            'totgovexpend',
            'domestbspend',
            'gftbcommit',
            'otherintltbcommit',
            'privatetbspend',
            'tbrelatedhealthcost',
            'socialmitigcost'
        ]
        self.filename = 'xls/data_input_4.xlsx'
        self.start_row = 0
        self.create_parlist = False
        self.horizontal = True

    def parse_row(self, row):
        raw_par = row[0]
        if raw_par == "":
            return
        self.i_par += 1
        self.par = self.parlist[self.i_par]
        self.data[self.par] = parse_year_data(row[1:], '')

    def get_data(self):
        return self.data


class ConstantsSheetReader:

    def __init__(self):
        self.subparlist = []
        self.data = {}
        self.subpar = None
        self.raw_subpar = None
        self.par = None
        self.i_par = -1
        self.i_subpar = -1
        self.name = 'constants'
        self.key = 'const'
        self.nested_parlist =  [
            [   'model_parameters', 
                [   'demo_rate_birth',
                    'demo_rate_death',
                    'epi_proportion_cases_smearpos',
                    'epi_proportion_cases_smearneg',
                    'epi_proportion_cases_extrapul',
                    'epi_proportion_cases',
                    'tb_multiplier_force_smearpos',
                    'tb_multiplier_force_smearneg',
                    'tb_multiplier_force_extrapul',
                    'tb_multiplier_force',
                    'tb_n_contact',
                    'tb_proportion_early_progression',
                    'tb_timeperiod_early_latent',
                    'tb_rate_late_progression',
                    'tb_proportion_casefatality_untreated_smearpos',
                    'tb_proportion_casefatality_untreated_smearneg',
                    'tb_proportion_casefatality_untreated',
                    'tb_timeperiod_activeuntreated',
                    'tb_multiplier_bcg_protection',
                    'program_prop_vac',
                    'program_prop_unvac',
                    'program_proportion_detect',
                    'program_algorithm_sensitivity',
                    'program_rate_start_treatment',
                    'tb_timeperiod_treatment_ds',
                    'tb_timeperiod_treatment_mdr',
                    'tb_timeperiod_treatment_xdr',
                    'tb_timeperiod_treatment_inappropriate',
                    'tb_timeperiod_infect_ontreatment_ds',
                    'tb_timeperiod_infect_ontreatment_mdr',
                    'tb_timeperiod_infect_ontreatment_xdr',
                    'tb_timeperiod_infect_ontreatment_inappropriate',
                    'program_proportion_success_ds',
                    'program_proportion_success_mdr',
                    'program_proportion_success_xdr',
                    'program_proportion_success_inappropriate',
                    'program_rate_restart_presenting',
                    'proportion_amplification',
                    'timepoint_introduce_mdr',
                    'timepoint_introduce_xdr',
                    'treatment_available_date',
                    'dots_start_date',
                    'finish_scaleup_date',
                    'pretreatment_available_proportion',
                    'dots_start_proportion',
                    'program_prop_assign_mdr',
                    'program_prop_assign_xdr',
                    'program_prop_lowquality',
                    'program_rate_leavelowquality',
                    'program_prop_nonsuccessoutcomes_death']], \
            [   'initials_for_compartments', 
                [   'susceptible_fully',
                    'latent_early', 
                    'latent_late', 
                    'active', 
                    'undertreatment']],\
            [   'disutility weights',
                [   'disutiuntxactivehiv',
                    'disutiuntxactivenohiv',
                    'disutitxactivehiv',
                    'disutitxactivehiv',
                    'disutiuntxlatenthiv',
                    'disutiuntxlatentnohiv',
                    'disutitxlatenthiv',
                    'disutitxlatentnohiv']]
        ]
        self.filename = 'xls/data_input_4.xlsx'
        self.start_row = 0
        self.create_parlist = False
        self.horizontal = True

    def parse_row(self, row):

        raw_par, raw_subpar = row[0:2]
        if raw_par != "":
            self.i_par += 1
            self.par, self.subparlist = self.nested_parlist[self.i_par]
            self.i_subpar = -1
            self.subpar = None
            self.data[self.par] = {}
            return
        if raw_subpar != "" and raw_subpar != self.raw_subpar:
            self.i_subpar += 1
            self.raw_subpar = raw_subpar
            self.subpar = self.subparlist[self.i_subpar]
        if raw_par == "" and raw_subpar == "":
            return
        best, low, high = replace_blanks(row[2:5], nan, '')
        self.data[self.par][self.subpar] = {
            'Best': best,
            'Low': low, 
            'High': high
        } 

    def get_data(self):
        return self.data


class NestedParamSheetReader:

    def __init__(self):
        self.subparlist = []
        self.data = {}
        self.subpar = None
        self.raw_subpar = None
        self.par = None
        self.i_par = -1
        self.i_subpar = -1
        self.name = 'XLS Sheet Name'
        self.key = 'data_key'
        self.nested_parlist = [
            [   'par0', 
                [   'subpar0', 
                    'subpar1'
                ]
            ], 
        ]
        self.filename = 'xls/data_input_4.xlsx'
        self.start_row = 0
        self.create_parlist = False
        self.horizontal = True

    def parse_row(self, row):
        raw_par, raw_subpar = row[0:2]
        if raw_par != "":
            self.i_par += 1
            self.par, self.subparlist = self.nested_parlist[self.i_par]
            self.i_subpar = -1
            self.subpar = None
            self.data[self.par] = {}
            return
        if raw_subpar != "" and raw_subpar != self.raw_subpar:
            self.i_subpar += 1
            self.raw_subpar = raw_subpar
            self.subpar = self.subparlist[self.i_subpar]
        if raw_par == "" and raw_subpar == "":
            return
        self.data[self.par][self.subpar] = parse_year_data(row[3:], '')

    def get_data(self):
        return self.data


class NestedParamWithRangeSheetReader:

    def __init__(self):
        self.subparlist = []
        self.data = {}
        self.subpar = None
        self.raw_subpar = None
        self.raw_par = None
        self.i_par = -1
        self.i_subpar = -1
        self.name = 'XLS Sheet Name'
        self.key = 'data_key'
        self.range = {
            'Best': [],
            'High': [],
            'Low': []
        }
        self.nested_parlist = [
            [
                'par0', 
                [
                    'subpar0', 
                    'subpar1'
                ]
            ], 
        ]
        self.filename = 'xls/data_input_4.xlsx'
        self.start_row = 0
        self.create_parlist = False
        self.horizontal = True

    def parse_row(self, row):
        raw_par, raw_subpar, blh = row[0:3]
        blh = str(blh)
        if raw_par != "":
            self.i_par += 1
            self.par, self.subparlist = self.nested_parlist[self.i_par]
            self.i_subpar = -1
            self.subpar = None
            self.data[self.par] = {}
            return
        if raw_subpar != "" and raw_subpar != self.raw_subpar:
            self.i_subpar += 1
            self.raw_subpar = raw_subpar
            self.subpar = self.subparlist[self.i_subpar]
            self.data[self.par][self.subpar] = copy.deepcopy(self.range)
        if blh == "":
            return
        self.data[self.par][self.subpar][blh] = parse_year_data(row[3:], '')

    def get_data(self):
        return self.data


class BcgCoverageSheetReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.name = 'BCG'
        self.key = 'bcg'
        self.parlist = []
        self.filename = 'xls/who_unicef_bcg_coverage.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 2
        self.horizontal = True

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]
        if row[2] == u'Cname':  # Year
            self.data[self.par] = []
            for i in range(4, len(row)):
                self.data[self.par] += [int(row[i])]
        else:  # Data
            self.data[self.par] =\
                parse_year_data(row[4:], '')
        # This sheet goes from 2014 backwards to 1980 from left to right, so:
        self.data[self.par] = list(reversed(self.data[self.par]))

    def get_data(self):
        return self.data


class BirthRateReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.name = 'Data'
        self.key = 'birth_rate'
        self.parlist = []
        self.filename = 'xls/world_bank_crude_birth_rate.xlsx'
        self.start_row = 0
        self.create_parlist = True
        self.column_for_keys = 2
        self.horizontal = True

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]
        if row[2] == u'Country Name':  # Year
            self.data[self.par] = []
            for i in range(4, len(row)):
                self.data[self.par] += [int(row[i][:4])]
        elif row[2] == '':  # Blank lines at the end
            return
        else:  # Data
            self.data[self.par] =\
                parse_year_data(row[4:], u'..')

    def get_data(self):
        return self.data


class LifeExpectancyReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.name = 'Data'
        self.key = 'life_expectancy'
        self.parlist = []
        self.filename = 'xls/world_bank_life_expectancy.xlsx'
        self.start_row = 3
        self.create_parlist = True
        self.column_for_keys = 0
        self.horizontal = True

    def parse_row(self, row):

        self.i_par += 1
        self.par = self.parlist[self.i_par]
        if row[0] == u'Country Name':  # Year
            self.data[self.par] = []
            for i in range(4, len(row)):
                self.data[self.par] += [int(row[i])]
        else:  # Data
            self.data[self.par] =\
                parse_year_data(row[4:], u'')

    def get_data(self):
        return self.data


class GlobalTbReportReader():

    def __init__(self):
        self.data = {}
        self.par = None
        self.i_par = -1
        self.name = 'TB_burden_countries_2016-04-19'
        self.key = 'tb'
        self.parlist = []
        self.filename = 'xls/TB_burden_countries_2016-04-19.xlsx'
        self.start_row = 1
        self.creat_parlist = True
        self.horizontal = False
        self.start_column = 0
        self.start_row = 1
        self.indices = {}

    def parse_col(self, col):

        if col[0] == u'country':
            for i in range(1, len(col)):
                self.indices[i] = col[i]
                if col[i] not in self.data:
                    self.data[col[i]] = {}
        else:
            for i in range(1, len(col)):
                if col[0] == u'year' or col[0] == u'iso_numeric':
                    item_to_add = int(col[i])
                else:
                    item_to_add = col[i]
                if col[0] not in self.data[self.indices[i]]:
                    self.data[self.indices[i]][col[0]] = []
                else:
                    self.data[self.indices[i]][col[0]] += [item_to_add]

    def get_data(self):

        return self.data


def read_xls_with_sheet_readers(sheet_readers=[]):

    result = {}
    for reader in sheet_readers:
        try:
            workbook = open_workbook(reader.filename)
        except:
            raise Exception('Failed to open spreadsheet: %s' % reader.filename)
        #print("Reading sheet \"{}\"".format(reader.name))
        sheet = workbook.sheet_by_name(reader.name)
        if reader.horizontal:
            for i_row in range(reader.start_row, sheet.nrows):
                if reader.create_parlist:
                    key = [sheet.row_values(i_row)[reader.column_for_keys]]
                    if key == [u'Cname'] or key == [u'Country Name']:
                        key = [u'year']
                    reader.parlist += key
                reader.parse_row(sheet.row_values(i_row))
        else:
            for i_col in range(reader.start_column, sheet.ncols):
                reader.parse_col(sheet.col_values(i_col))
        result[reader.key] = reader.get_data()
    return result


def read_input_data_xls():

    sheet_readers = []

    # population_sheet_reader = NestedParamWithRangeSheetReader()
    # population_sheet_reader.name = 'population_size'
    # population_sheet_reader.key = 'popsize'
    # population_sheet_reader.nested_parlist =  [
    #     [   'Population size',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov'
    #         ]
    #     ]
    # ]
    # sheet_readers.append(population_sheet_reader)

    # tb_prevalence_sheet_reader = NestedParamWithRangeSheetReader()
    # tb_prevalence_sheet_reader.name = 'TB prevalence'
    # tb_prevalence_sheet_reader.key = 'tbprev'
    # tb_prevalence_sheet_reader.nested_parlist =  [
    #     [   '0_4yr',
    #         [   'ds_04yr',
    #             'mdr_04yr',
    #             'xdr_04yr'
    #         ]
    #     ],
    #     [   '5_14yr',
    #         [   'ds_514yr',
    #             'mdr_514yr',
    #             'xdr_514yr'
    #         ]
    #     ],
    #     [   '15abov',
    #         [   'ds_15abov',
    #             'mdr_15abov',
    #             'xdr_15abov'
    #         ]
    #     ]
    # ]
    # sheet_readers.append(tb_prevalence_sheet_reader)
    #
    # tb_incidence_sheet_reader = NestedParamWithRangeSheetReader()
    # tb_incidence_sheet_reader.name = 'TB incidence'
    # tb_incidence_sheet_reader.key = 'tbinc'
    # tb_incidence_sheet_reader.nested_parlist =  [
    #     [   '0_4yr',
    #         [   'ds_04yr',
    #             'mdr_04yr',
    #             'xdr_04yr'
    #         ]
    #     ],
    #     [   '5_14yr',
    #         [   'ds_514yr',
    #             'mdr_514yr',
    #             'xdr_514yr'
    #         ]
    #     ],
    #     [   '15abov',
    #         [   'ds_15abov',
    #             'mdr_15abov',
    #             'xdr_15abov'
    #         ]
    #     ]
    # ]
    # sheet_readers.append(tb_incidence_sheet_reader)
    #
    # comorbidity_sheet_reader = NestedParamWithRangeSheetReader()
    # comorbidity_sheet_reader.name = 'comorbidity'
    # comorbidity_sheet_reader.key = 'comor'
    # comorbidity_sheet_reader.nested_parlist =  [
    #     [   'malnutrition',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov',
    #             'aggregate'
    #         ]
    #     ],
    #     [   'diabetes',
    #         [  '04yr',
    #             '5_14yr',
    #             '15abov',
    #             'aggregate'
    #         ]
    #     ],
    #     [   'HIV',
    #         [   '04yr_CD4_300',
    #             '04yr_CD4_200_300',
    #             '04yr_CD4_200',
    #             '04yr_aggregate',
    #             '5_14yr_CD4_300',
    #             '5_14yr_CD4_200_300',
    #             '5_14yr_CD4_200',
    #             '5_14yr_aggregate',
    #             '15abov_CD4_300',
    #             '15abov_CD4_200_300',
    #             '15abov_CD4_200',
    #             '15abov_aggregate'
    #         ]
    #     ]
    # ]
    # sheet_readers.append(comorbidity_sheet_reader)
    #
    # cost_coverage_sheet_reader = NestedParamWithRangeSheetReader()
    # cost_coverage_sheet_reader.name = 'cost and coverage'
    # cost_coverage_sheet_reader.key = 'costcov'
    # cost_coverage_sheet_reader.range = {'Coverage':[], 'Cost':[]}
    # cost_coverage_sheet_reader.nested_parlist =  [
    #     [   'Cost and coverage',
    #         [   'Active and intensified case finding',
    #             'Treatment of active TB',
    #             'Preventive therapy for latent TB',
    #             'Vaccination',
    #             'Patient isolation',
    #             'Drug susceptibility testing',
    #             'Preventive therapy for patients with HIV co-infection',
    #             'Infection control in healthcare facilities',
    #         ]
    #     ]
    # ]
    # sheet_readers.append(cost_coverage_sheet_reader)
    #
    # testing_treatment_sheet_reader = NestedParamSheetReader()
    # testing_treatment_sheet_reader.name = 'testing_treatment'
    # testing_treatment_sheet_reader.key = 'testtx'
    # testing_treatment_sheet_reader.nested_parlist =  [
    #     [   '%testedactiveTB',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov']], \
    #     [   '%testedlatentTB',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov']],\
    #     [   '%testedsuscept',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov']],\
    #     [   'numberinittxactiveTB',
    #         [   '04yr_DSregimen',
    #             '04yr_MDRregimen',
    #             '04yr_XDRregimen',
    #             '5_14yr_DSregimen',
    #             '5_14yr_MDRregimen',
    #             '5_14yr_XDRregimen',
    #             '15abov_DSregimen',
    #             '15abov_MDRregimen',
    #             '15abov_XDRregimen']],\
    #     [   'numbercompletetxactiveTB',
    #         [   '04yr_DSregimen',
    #             '04yr_MDRregimen',
    #             '04yr_XDRregimen',
    #             '5_14yr_DSregimen',
    #             '5_14yr_MDRregimen',
    #             '5_14yr_XDRregimen',
    #             '15abov_DSregimen',
    #             '15abov_MDRregimen',
    #             '15abov_XDRregimen']],\
    #     [   'numberinittxlatentTB',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov']],\
    #     ['numbercompletetxlatentTB',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov']]
    # ]
    # sheet_readers.append(testing_treatment_sheet_reader)
    #
    # other_epidemiology_sheet_reader = NestedParamSheetReader()
    # other_epidemiology_sheet_reader.name = 'other_epidemiology'
    # other_epidemiology_sheet_reader.key = 'otherepi'
    # other_epidemiology_sheet_reader.nested_parlist = [
    #     [   '%died_nonTB',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov'
    #         ]
    #     ],
    #     [   '%died_TBrelated',
    #         [   '04yr',
    #             '5_14yr',
    #             '15abov'
    #         ]
    #     ],
    #     [   'birthrate',
    #         [   'birthrate']
    #     ],
    # ]
    #
    # sheet_readers.append(other_epidemiology_sheet_reader)
    # sheet_readers.append(ConstantsSheetReader())
    # sheet_readers.append(MacroeconomicsSheetReader())
    # sheet_readers.append(BcgCoverageSheetReader())
    # sheet_readers.append(BirthRateReader())
    # sheet_readers.append(LifeExpectancyReader())
    sheet_readers.append(GlobalTbReportReader())

    return read_xls_with_sheet_readers(sheet_readers)


if __name__ == "__main__":
    import json
    data = read_input_data_xls()  # C:\Users\ntdoan\Github\AuTuMN\autumn\xls
    open('spreadsheet.out.txt', 'w').write(json.dumps(data, indent=2))
    print(data)


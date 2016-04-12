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


def replace_blanks(a_list, new_val):
    return [new_val if val == '' else val for val in a_list]


def parse_year_data(these_data):
    these_data = replace_blanks(these_data, nan)
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

    def parse_row(self, row):
        raw_par = row[0]
        if raw_par == "":
            return
        self.i_par += 1
        self.par = self.parlist[self.i_par]
        self.data[self.par] = parse_year_data(row[1:])

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
        best, low, high = replace_blanks(row[2:5], nan)
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
        self.data[self.par][self.subpar] = parse_year_data(row[3:])

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
        self.data[self.par][self.subpar][blh] = parse_year_data(row[3:])

    def get_data(self):
        return self.data



def read_xls_with_sheet_readers(filename, sheet_readers=[]):
    try: 
        workbook = open_workbook(filename) 
    except: 
        raise Exception('Failed to open spreadsheet: %s' % filename)
    result = {}
    for reader in sheet_readers:
        #print("Reading sheet \"{}\"".format(reader.name))
        sheet = workbook.sheet_by_name(reader.name)
        for i_row in range(sheet.nrows):
            reader.parse_row(sheet.row_values(i_row))                
        result[reader.key] = reader.get_data()
    return result



def read_input_data_xls(filename):

    sheet_readers = []

    population_sheet_reader = NestedParamWithRangeSheetReader()
    population_sheet_reader.name = 'population_size'
    population_sheet_reader.key = 'popsize'
    population_sheet_reader.nested_parlist =  [
        [   'Population size',
            [   '04yr',
                '5_14yr',
                '15abov'
            ]
        ]        
    ]
    sheet_readers.append(population_sheet_reader)

    tb_prevalence_sheet_reader = NestedParamWithRangeSheetReader()
    tb_prevalence_sheet_reader.name = 'TB prevalence'
    tb_prevalence_sheet_reader.key = 'tbprev'
    tb_prevalence_sheet_reader.nested_parlist =  [
        [   '0_4yr', 
            [   'ds_04yr', 
                'mdr_04yr', 
                'xdr_04yr'
            ]
        ], 
        [   '5_14yr', 
            [   'ds_514yr', 
                'mdr_514yr', 
                'xdr_514yr'
            ]
        ],
        [   '15abov',
            [   'ds_15abov',
                'mdr_15abov',
                'xdr_15abov'
            ]
        ]
    ]
    sheet_readers.append(tb_prevalence_sheet_reader)

    tb_incidence_sheet_reader = NestedParamWithRangeSheetReader()
    tb_incidence_sheet_reader.name = 'TB incidence'
    tb_incidence_sheet_reader.key = 'tbinc'
    tb_incidence_sheet_reader.nested_parlist =  [
        [   '0_4yr', 
            [   'ds_04yr', 
                'mdr_04yr', 
                'xdr_04yr'
            ]
        ], 
        [   '5_14yr', 
            [   'ds_514yr', 
                'mdr_514yr', 
                'xdr_514yr'
            ]
        ],
        [   '15abov',
            [   'ds_15abov',
                'mdr_15abov',
                'xdr_15abov'
            ]
        ]
    ]        
    sheet_readers.append(tb_incidence_sheet_reader)
   
    comorbidity_sheet_reader = NestedParamWithRangeSheetReader()
    comorbidity_sheet_reader.name = 'comorbidity'
    comorbidity_sheet_reader.key = 'comor'
    comorbidity_sheet_reader.nested_parlist =  [
        [   'malnutrition', 
            [   '04yr', 
                '5_14yr', 
                '15abov',
                'aggregate'
            ]
        ], 
        [   'diabetes', 
            [  '04yr', 
                '5_14yr', 
                '15abov',
                'aggregate'
            ]
        ],
        [   'HIV',
            [   '04yr_CD4_300',
                '04yr_CD4_200_300',
                '04yr_CD4_200',
                '04yr_aggregate',
                '5_14yr_CD4_300',
                '5_14yr_CD4_200_300', 
                '5_14yr_CD4_200', 
                '5_14yr_aggregate', 
                '15abov_CD4_300',
                '15abov_CD4_200_300', 
                '15abov_CD4_200', 
                '15abov_aggregate'
            ]
        ]
    ]
    sheet_readers.append(comorbidity_sheet_reader)

    cost_coverage_sheet_reader = NestedParamWithRangeSheetReader()
    cost_coverage_sheet_reader.name = 'cost and coverage'
    cost_coverage_sheet_reader.key = 'costcov'
    cost_coverage_sheet_reader.range = {'Coverage':[], 'Cost':[]}
    cost_coverage_sheet_reader.nested_parlist =  [
        [   'Cost and coverage',
            [   'Active and intensified case finding', 
                'Treatment of active TB', 
                'Preventive therapy for latent TB',
                'Vaccination',
                'Patient isolation',
                'Drug susceptibility testing',
                'Preventive therapy for patients with HIV co-infection',
                'Infection control in healthcare facilities',
            ]
        ]
    ]
    sheet_readers.append(cost_coverage_sheet_reader)

    testing_treatment_sheet_reader = NestedParamSheetReader()
    testing_treatment_sheet_reader.name = 'testing_treatment'
    testing_treatment_sheet_reader.key = 'testtx'
    testing_treatment_sheet_reader.nested_parlist =  [
        [   '%testedactiveTB', 
            [   '04yr', 
                '5_14yr', 
                '15abov']], \
        [   '%testedlatentTB', 
            [   '04yr', 
                '5_14yr', 
                '15abov']],\
        [   '%testedsuscept',
            [   '04yr',
                '5_14yr',
                '15abov']],\
        [   'numberinittxactiveTB',
            [   '04yr_DSregimen',
                '04yr_MDRregimen',
                '04yr_XDRregimen',
                '5_14yr_DSregimen',
                '5_14yr_MDRregimen',
                '5_14yr_XDRregimen',
                '15abov_DSregimen',
                '15abov_MDRregimen',
                '15abov_XDRregimen']],\
        [   'numbercompletetxactiveTB',
            [   '04yr_DSregimen',
                '04yr_MDRregimen',
                '04yr_XDRregimen',
                '5_14yr_DSregimen',
                '5_14yr_MDRregimen',
                '5_14yr_XDRregimen',
                '15abov_DSregimen',
                '15abov_MDRregimen',
                '15abov_XDRregimen']],\
        [   'numberinittxlatentTB',
            [   '04yr',
                '5_14yr',
                '15abov']],\
        ['numbercompletetxlatentTB',
            [   '04yr',
                '5_14yr',
                '15abov']]      
    ]
    sheet_readers.append(testing_treatment_sheet_reader)
   
    other_epidemiology_sheet_reader = NestedParamSheetReader()
    other_epidemiology_sheet_reader.name = 'other_epidemiology'
    other_epidemiology_sheet_reader.key = 'otherepi'
    other_epidemiology_sheet_reader.nested_parlist = [
        [   '%died_nonTB', 
            [   '04yr', 
                '5_14yr', 
                '15abov'
            ]
        ], 
        [   '%died_TBrelated', 
            [   '04yr', 
                '5_14yr', 
                '15abov'
            ]
        ],
        [   'birthrate',
            [   'birthrate']
        ],
    ]
    sheet_readers.append(other_epidemiology_sheet_reader)

    sheet_readers.append(ConstantsSheetReader())

    sheet_readers.append(MacroeconomicsSheetReader())
    
    return read_xls_with_sheet_readers(filename, sheet_readers)


if __name__ == "__main__":
    import json
    data = read_input_data_xls('xls/data_input_4.xlsx')
    open('spreadsheet.out.txt', 'w').write(json.dumps(data, indent=2))
    #print(data)


            
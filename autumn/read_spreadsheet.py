# -*- coding: utf-8 -*-

from __future__ import print_function
from xlrd import open_workbook # For opening Excel workbooks
from numpy import nan, zeros, isnan, array, logical_or, nonzero # For reading in empty values

"""
Import model inputs from Excel spreadsheet 
Version: 21 November 2015 by Tan Doan 
"""
"""
TO DO LIST 
1. UNCERTAINTY RANGE; only able to read the HIGH row for variables that have High, Best and Low inputs (Workbooks: population size,
TB prevalence, TB incidence, comorbidity)
2. FIX MACROECONOMICS workbook: not able to read any data in this workbook

"""
def get_input (filename, verbose=0):
    
 
    constants = \
    [
        [
        'constants', 'const', 
        [
          ['model_parameters', 
               ['rate_pop_birth', 
                'rate_pop_death', 
                'n_tbfixed_contact', 
                'rate_tbfixed_earlyprog', 
                'rate_tbfixed_lateprog', 
                'rate_tbfixed_stabilise', 
                'rate_tbfixed_recover', 
                'rate_tbfixed_death', 
                'rate_tbprog_detect']], \
          ['initials_for_compartments', 
               ['susceptible', 
                'latent_early', 
                'latent_late', 
                'active', 
                'undertreatment']],\
          ['disutility weights',
              ['disutiuntxactivehiv',
              'disutiuntxactivenohiv',
              'disutitxactivehiv',
              'disutitxactivehiv',
              'disutiuntxlatenthiv',
              'disutiuntxlatentnohiv',
              'disutitxlatenthiv',
              'disutitxlatentnohiv']]
        ]
      ]
   ]
   
    macroeconomics = \
   [
       ['macroeconomics', 'macro',
            ['cpi',
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
             'socialmitigcost']        
        ]
   ]
    
    costcoverage = \
    [
        ['cost and coverage', 'costcov',
             ['cov',
              'cost']
        ]
    ]
    
    tbprevalence = \
    [
        [
        'TB prevalence', 'tbprev', 
        [
          ['0_4yr', 
               ['ds_04yr', 
                'mdr_04yr', 
                'xdr_04yr']], \
          ['5_14yr', 
               ['ds_514yr', 
                'mdr_514yr', 
                'xdr_514yr']],\
          ['15abov',
              ['ds_15abov',
              'mdr_15abov',
              'xdr_15abov']]
        ]
      ]
   ]
   
    tbincidence = \
    [
        [
        'TB incidence', 'tbinc', 
        [
          ['0_4yr', 
               ['ds_04yr', 
                'mdr_04yr', 
                'xdr_04yr']], \
          ['5_14yr', 
               ['ds_514yr', 
                'mdr_514yr', 
                'xdr_514yr']],\
          ['15abov',
              ['ds_15abov',
              'mdr_15abov',
              'xdr_15abov']]
        ]
      ]
   ]        
   
    
    comorbidity = \
    [
        [
        'comorbidity', 'comor', 
        [
          ['malnutrition', 
               ['04yr', 
                '5_14yr', 
                '15abov',
                'aggregate']], \
          ['diabetes', 
               ['04yr', 
                '5_14yr', 
                '15abov',
                'aggregate']],\
          ['HIV',
              ['04yr_CD4_300',
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
              '15abov_aggregate']]
        ]
      ]
   ]        
    
      
    population_size = \
   [
       ['population_size', 'popsize',
            ['04yr',
            '5_14yr',
            '15abov']        
        ]
   ]
   
   
    testing_treatment = \
   [
        [
        'testing_treatment', 'testtx', 
        [
          ['%testedactiveTB', 
               ['04yr', 
                '5_14yr', 
                '15abov']], \
          ['%testedlatentTB', 
               ['04yr', 
                '5_14yr', 
                '15abov']],\
          ['%testedsuscept',
              ['04yr',
              '5_14yr',
              '15abov']],\
          ['numberinittxactiveTB',
               ['04yr_DSregimen',
                '04yr_MDRregimen',
                '04yr_XDRregimen',
                '5_14yr_DSregimen',
                '5_14yr_MDRregimen',
                '5_14yr_XDRregimen',
                '15abov_DSregimen',
                '15abov_MDRregimen',
                '15abov_XDRregimen']],\
          ['numbercompletetxactiveTB',
               ['04yr_DSregimen',
                '04yr_MDRregimen',
                '04yr_XDRregimen',
                '5_14yr_DSregimen',
                '5_14yr_MDRregimen',
                '5_14yr_XDRregimen',
                '15abov_DSregimen',
                '15abov_MDRregimen',
                '15abov_XDRregimen']],\
          ['numberinittxlatentTB',
               ['04yr',
                '5_14yr',
                '15abov']],\
          ['numbercompletetxlatentTB',
               ['04yr',
                '5_14yr',
                '15abov']]      
        ]
      ]
   ]    
 
    other_epidemiology = \
   [
        [
        'other_epidemiology', 'otherepi', 
        [
          ['%died_nonTB', 
               ['04yr', 
                '5_14yr', 
                '15abov']], \
          ['%died_TBrelated', 
               ['04yr', 
                '5_14yr', 
                '15abov']],\
          ['birthrate',
              ['birthrate']],\
        ]
      ]
   ]    
 
 
    
    sheetstructure = dict()
    sheetstructure['constants'] = constants
    sheetstructure['macroeconomics'] = macroeconomics
    sheetstructure['costcoverage'] = costcoverage
    sheetstructure['tbprevalence'] = tbprevalence
    sheetstructure['tbincidence'] = tbincidence
    sheetstructure['comorbidity'] = comorbidity
    sheetstructure['population_size'] = population_size
    sheetstructure['testing_treatment'] = testing_treatment 
    sheetstructure['other_epidemiology'] = other_epidemiology  
    
    

    ###########################################################################
    ## Load data sheet
    ###########################################################################
    
    data = dict() # Create structure for holding data
    try: workbook = open_workbook(filename) # Open workbook
    except: raise Exception('Failed to load spreadsheet: file "%s" not found!' % filename)
    
    sheetstructure_keys = list(sheetstructure.keys())
    
    # Keys: This method returns a list of all the available keys in the dictionary
    # for example: dict = {'Name': 'Zara', 'Age': 7}  print "Value : %s" %  dict.keys() --> Value : ['Age', 'Name']

    ## Loop over each group of sheets
    for groupname in sheetstructure_keys: # Loop over each type of data
        sheetgroup = sheetstructure[groupname]
        for sheet in sheetgroup: # Loop over each workbook for that data
            lastdatacol = None    
            # Name of the workbook
            sheetname = sheet[0] 
            name = sheet[1] # Pull out the name of this field
            subparlist = sheet[2] # List of subparameters
            #if verbose == 0:
               # print (subparlist)
            data[name] = dict() # Create structure for holding data
            sheetdata = workbook.sheet_by_name(sheetname) # Load this workbook
            parcount = -1 # Initialize the parameter count
                            
           ## Calculate columns for which data are entered, and store the year ranges     
            
                               
            if name == 'macro': # Need to gather year ranges for economic data
                data['epiyears'] = [] # Initialize epidemiology data years
                for col in range(sheetdata.ncols):
                    thiscell = sheetdata.cell_value(0, col) # 0 is the 1st row which is where the year data should be
                    if thiscell =='' and len(data['epiyears'])>0: #  We've gotten to the end
                        lastdatacol = col # Store this column number
                        break # Quit
                    elif thiscell != '': # Nope, more years, keep going
                        data['epiyears'].append(float(thiscell)) # Add this year
                        
                        
            if name == 'tbprev' or name == 'tbinc' or name == 'comor': # Need to gather year ranges for epidemiology data
                data['epiyears'] = [] # Initialize epidemiology data years
                for col in range(sheetdata.ncols):
                    thiscell = sheetdata.cell_value(0, col) # 0 is the 1st row which is where the year data should be
                    if thiscell =='' and len(data['epiyears'])>0: #  We've gotten to the end
                        lastdatacol = col # Store this column number
                        break # Quit
                    elif thiscell != '': # Nope, more years, keep going
                        data['epiyears'].append(float(thiscell)) # Add this year
                        
                        
            if lastdatacol:  
                assumptioncol = lastdatacol + 1 # The "OR" space is in between

            # Loop over each row in the workbook
            for row in range(sheetdata.nrows): 
                paramcategory = sheetdata.cell_value(row,0) # See what's in the first column for this row
                if paramcategory != '': # It's not blank: e.g. "Model parameters"
                    parcount += 1 # Increment the parameter count
                    
                    if groupname == 'constants':
                        thispar = subparlist[parcount][0] # Get the name of this parameter e.g. "model_parameters"
                        data[name][thispar] = dict() # Need another structure 
#                                            
                    if groupname == 'macroeconomics':
                        thispar = subparlist[parcount] # Get the name of this parameter e.g. "GDP"
                        data[name][thispar] = dict() # Need another structure
                                                    
                    if groupname == 'costcoverage':
                        data[name][subparlist[0]] = [] # Initialize coverage to an empty list -- i.e. data['costcov'].cov
                        data[name][subparlist[1]] = [] # Initialize cost to an empty list -- i.e. data['costcov']['cost']
                        
                    if groupname == 'tbprevalence' or groupname == 'tbincidence':
                          thispar = subparlist[parcount][0] # Get the name of this parameter, e.g. '0-4 years'
                          data[name][thispar] = dict() # Initialize to empty list 
                                                    
                    if groupname == 'comorbidity':
                        thispar = subparlist[parcount][0] # Get the name of this parameter e.g. "Malnutrition prevalence"
                        data[name][thispar] = dict() # Need another structure 
#                        
                    if groupname == 'population_size':
                        thispar = subparlist[parcount] # Get the name of this parameter e.g. "0-4 years"
                        data[name][thispar] =[] # Need another structure 
                        
                    if groupname == 'testing_treatment':
                        thispar = subparlist[parcount][0] # Get the name of this parameter e.g. "model_parameters"
                        data[name][thispar] = dict() # Need another structure 
#                                                
                    if groupname == 'other_epidemiology':
                        thispar = subparlist[parcount][0] # Get the name of this parameter e.g. "model_parameters"
                        data[name][thispar] = dict() # Need another structure 
#                            
                subparam = ''

                if paramcategory == '': # The first column is blank: it's time for the data
                    subparam = sheetdata.cell_value(row, 1) # Get the name of a subparameter, e.g. "rate_pop_birth"
                    
                    if subparam != '': # The subparameter name isn't blank, load something!
                       #create a new dictionary entry
                        if groupname =='constants':
                            thesedata = sheetdata.row_values(row, start_colx=2, end_colx=5) # Data starts in 3rd column, finishes in 5th column
                            thesedata = list(map(lambda val: None if val=='' else val, thesedata)) # Replace blanks with None, in case we don't have uncertainty range
                            subpar = subparlist[parcount][1].pop(0) # Pop first entry of subparameter list
                            data[name][thispar][subpar] = thesedata # Store data
                            
                    
                        if groupname =='testing_treatment':
                            thesedata = sheetdata.row_values(row, start_colx=2, end_colx=lastdatacol) # Data starts in 3rd column, finishes in 5th column
                            thesedata = list(map(lambda val: None if val=='' else val, thesedata)) # Replace blanks with None, in case we don't have uncertainty range
                            subpar = subparlist[parcount][1].pop(0) # Pop first entry of subparameter list
                            data[name][thispar][subpar] = thesedata # Store data
                            
                        if groupname =='other_epidemiology':
                            thesedata = sheetdata.row_values(row, start_colx=2, end_colx=lastdatacol) # Data starts in 3rd column, finishes in 5th column
                            thesedata = list(map(lambda val: None if val=='' else val, thesedata)) # Replace blanks with None, in case we don't have uncertainty range
                            subpar = subparlist[parcount][1].pop(0) # Pop first entry of subparameter list
                            data[name][thispar][subpar] = thesedata # Store data
                                                
                        if groupname == 'macroeconomics':
                            thesedata = sheetdata.row_values(row, start_colx=1, end_colx=lastdatacol) # Data starts in 2nd column
                            thesedata = list(map(lambda val: nan if val=='' else val, thesedata)) # Replace blanks with nan
                            #data[name][thispar].append(thesedata)

                        if groupname=='costcoverage':
                            thesedata = sheetdata.row_values(row, start_colx=3, end_colx=lastdatacol) # Data starts in 4th column
                            thesedata = list(map(lambda val: nan if val=='' else val, thesedata)) # Replace blanks with nan
                            assumptiondata = sheetdata.cell_value(row, assumptioncol)
                            if assumptiondata != '': thesedata = [assumptiondata] # Replace the (presumably blank) data if a non-blank assumption has been entered
                            ccindices = {'Coverage':0, 'Cost':1} # Define best-low-high indices
                            cc = sheetdata.cell_value(row, 2) # Read in whether indicator is best, low, or high
                            data[name][subparlist[ccindices[cc]]].append(thesedata) # Actually append the data
                            
                        if groupname in ['tbprevalence', 'tbincidence', 'comorbidity']:
                            
                            #if verbose == 0: 
                                #print(data, name, thispar)
                                #print(x)
#                            
#                            if len(data[name][thispar]) == 0: 
#                                data[name][thispar][subpar] = [[] for z in range(3)] # Create new variable for high best low 
#                                x = data[name][thispar]
##                                if verbose == 0:
#                                    print(x)
                                    
                            thesedata = sheetdata.row_values(row, start_colx=3, end_colx=lastdatacol) # Data starts in 4th column
                            thesedata = list(map(lambda val: nan if val== '' else val, thesedata)) # Replace blanks with nan
                            subpar = subparlist[parcount][1].pop(0) # Pop first entry of subparameter list
                            assumptiondata = sheetdata.cell_value(row, assumptioncol)
                            if assumptiondata != '': thesedata = [assumptiondata] # Replace the (presumably blank) data if a non-blank assumption has been entered
         #                   blhindices = {'Best':0, 'Low': 1, 'High': 2} # Define best-low-high indices
                            #if verbose == 0: 
                             #   print (blhindices)
          #                  blh = sheetdata.cell_value(row, 2) # Read in whether indicator is best, low, or high
                                                      
   #                         data[name][thispar][subpar][blhindices[blh]].append(thesedata) # Actually append the data
                            data[name][thispar][subpar] = thesedata # Store data
                                                                               
                        if groupname == 'population_size':                            
                            thesedata = sheetdata.row_values(row, start_colx=3, end_colx=lastdatacol) # Data starts in 4th column
                            thesedata = list(map(lambda val: nan if val== '' else val, thesedata)) # Replace blanks with nan
                            #subpar = subparlist[parcount][1].pop(0) # Pop first entry of subparameter list
                            assumptiondata = sheetdata.cell_value(row, assumptioncol)
                            if assumptiondata != '': thesedata = [assumptiondata] # Replace the (presumably blank) data if a non-blank assumption has been entered
         #                   blhindices = {'Best':0, 'Low': 1, 'High': 2} # Define best-low-high indices                      
          #                  blh = sheetdata.cell_value(row, 2) # Read in whether indicator is best, low, or high
                                                      
   #                         data[name][thispar][subpar][blhindices[blh]].append(thesedata) # Actually append the data
                            #data[name][thispar][subpar] = thesedata # Store data
                            data[name][thispar] = thesedata # Store data
                    
    if verbose > 0:
        print('...done loading data.', 2, verbose)
 
    return data 


if __name__ == "__main__":
    data = get_input('data_input_3.xlsx', verbose=0)
    print(data)

            
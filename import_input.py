# -*- coding: utf-8 -*-
"""
Import model inputs from Excel spreadsheet 
Version: 21 November 2015 by Tan Doan 
"""

def import_input (filename='model_input.xlsx', verbose=0):
    
    from utils import printv
    from xlrd import open_workbook # For opening Excel workbooks
 
    # Constants -- array sizes are scalars x uncertainty
    constants = [
                 ['constants', 'const', [['model_parameters', ['rate_pop_birth', 'rate_pop_death', 'n_tbfixed_contact', 'rate_tbfixed_earlyprog', 'rate_tbfixed_lateprog', 'rate_tbfixed_stabilise', 'rate_tbfixed_recover', 'rate_tbfixed_death', 'rate_tbprog_detect']], \
                                       ['initials_for_compartments', ['susceptible', 'latent_early', 'latent_late', 'active', 'undertreatment']]]]
                ]
     
    ## Allows the list of groups to be used as name and also as variables
    ## This may be useful in the future when we have multiple sheets 
    sheetstructure = dict()
    sheetstructure['constants'] = constants

    ###########################################################################
    ## Load data sheet
    ###########################################################################
    
    data = dict() # Create structure for holding data
    try: workbook = open_workbook(filename) # Open workbook
    except: raise Exception('Failed to load spreadsheet: file "%s" not found!' % filename)
    
    sheetstructure_keys = list(sheetstructure.keys())

    ## Loop over each group of sheets
    for groupname in sheetstructure_keys: # Loop over each type of data
        sheetgroup = sheetstructure[groupname]
        for sheet in sheetgroup: # Loop over each workbook for that data
            lastdatacol = None
            sheetname = sheet[0] # Name of the workbook
            name = sheet[1] # Pull out the name of this field
            subparlist = sheet[2] # List of subparameters
            data[name] = dict() # Create structure for holding data
            sheetdata = workbook.sheet_by_name(sheetname) # Load this workbook
            parcount = -1 # Initialize the parameter count
            printv('Loading "%s"...' % sheetname, 2, verbose)
            
            # Loop over each row in the workbook
            for row in range(sheetdata.nrows): 
                paramcategory = sheetdata.cell_value(row,0) # See what's in the first column for this row
                if paramcategory != '': # It's not blank: e.g. "Model parameters"
                    printv('Loading "%s"...' % paramcategory, 3, verbose)
                    parcount += 1 # Increment the parameter count
                    
                    if groupname=='constants':
                        thispar = subparlist[parcount][0] # Get the name of this parameter e.g. "model_parameters"
                        data[name][thispar] = dict() # Need another structure   
                    else:
                        raise Exception('Group name %s not recognized!' % groupname)
                        
                                        
                if paramcategory == '': # The first column is blank: it's time for the data
                    subparam = sheetdata.cell_value(row, 1) # Get the name of a subparameter, e.g. "rate_pop_birth"
                    
                if subparam != '': # The subparameter name isn't blank, load something!
                    printv('Parameter: %s' % subparam, 4, verbose)
                    
                       #create a new dictionary entry
                
                    thesedata = sheetdata.row_values(row, start_colx=3, end_colx=5) # Data starts in 3rd column, finishes in 5th column
                    thesedata = list(map(lambda val: nan if val=='' else val, thesedata)) # Replace blanks with nan, in case we don't have uncertainty range
                    subpar = subparlist[parcount][1].pop(0) # Pop first entry of subparameter list
                    data[name][thispar][subpar] = thesedata # Store data

    printv('...done loading data.', 2, verbose)
 
    return data 

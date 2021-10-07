### COVID-19 Australian data

Collated publicly available COVID-19 data for Australia from https://www.covid19data.com.au/

Currently used to track testing. See their [data notes](https://www.covid19data.com.au/) for more details. N.B some negative values represent revisions to the data.

Data is pulled from Matt Bolton's COVID-19 data GitHub [here](https://github.com/M3IT/COVID-19_Data)

'lga_test.csv' - one time file provided by James for VIC paper.

### DHHS provides the 

[DHHS sharepoint](https://dhhsvicgovau.sharepoint.com/sites/SDE3-PHIDE/Shared%20Documents/Documents/Health%20service%20and%20mortality%20forecast?e=5%3a72cf5a5950dc4aa1b06a8a329a8fc4b6&at=9)

Select the following files with the date prefix DDMMYYY as a zip then unzip. 
16092021_nAdmissions_by_AdmissionDate_LGA_Age_Acquired_AdmittedToICU_Ventilated.csv
16092021_nEncounters_by_EncounterDate_Postcode_AgeGroup_isPfizer_DoseNumber.csv
16092021_NewCases_by_DiagnosisDate_LGA_Age_Acquired.csv

Run script/input_targets_dhhs.py


[CHRIS website](https://chris.health.gov.au/) download 'COVID-19' report and rename as monitoringreport.secert.csv

Then run /scripts/input_targets_dhhs.py. This will update victoria_2021 project targets and vac_cov.csv (current fodler)
Finally, after ensuring all other project application requirement are met do a inputs.db build to ingest the vac_cov.csv.

'postcode lphu concordance.csv' - provided by Timothy Cameron (Health) <timothy.cameron@health.vic.gov.au> on 9th Sep 2021 by email.

'yougov_australia.csv' data obtained from [YouGov](https://github.com/YouGov-Data/covid-19-tracker/blob/master/data/australia.zip)
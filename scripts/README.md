This readme documents the DHHS data update process.

1. Download the CSV from DHHS's secure folder to your local  AuTuMN repo folder data/inputs.
2. Rename the file to 'monashmodelextract.secret.csv'
3. Download the CSV report from CHRIS website to your local  AuTuMN repo folder data/inputs.
4. Rename the file to 'monitoringreport.secret.csv'
3. Run the dhhs_data_upload.py script located at \scripts

This will update a target.secret.json file for each region and generate a targets.encrypted.json file. 
The files are located in two folders, data/inputs/imports and apps/regions/<\region name>.

Select the all the '<>.encrypted.json' file and  data/secret-hashes.json and push to GitHub.

This readme documents the Philippines target calibration data update process.
The functions in phl_data_upload.py do the following:
1. Downloads data
2. Formats and filters data by region (using a duplicated dataset for the national model)
3. Calculated daily ICU occupancy, confirmed cases, and cumulative deaths
4. Dumps calibration data (step 3) into json files
5. Deletes files 

Run the following lines of code in order to update:
fetch_phl_data()
fassster_filename = fassster_data_filepath()
rename_regions(PHL_doh_dest, "region", "NATIONAL CAPITAL REGION (NCR)", "REGION IV-A (CALABAR ZON)", "REGION VII (CENTRAL VISAYAS)")
rename_regions(fassster_filename, "Region", "NCR", "4A", "07")
duplicate_data(PHL_doh_dest, "region")
duplicate_data(fassster_filename, "Region")
filter_df_by_regions(PHL_doh_dest, "region")
filter_df_by_regions(fassster_filename, "Region")
process_icu_data()
process_accumulated_death_data(fassster_filename)
process_notifications_data(fassster_filename)
update_calibration_phl()
remove_files(fassster_filename)
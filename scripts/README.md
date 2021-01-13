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

1. Update \autumun\inputs\covid_phl\fetch.py (Testing numbers - do this before a rebuild of inputs.db)
The daily data link is available from https://drive.google.com/drive/folders/1ZPPcVU4M7T-dtRyUceb0pMAd8ickYf8o.
On page 5 of the Readme there will be a link to the daily data folder e.g. Link to DOH Data Drop (09/08): https://bit.ly/2F8oypc.
*alternate link DoH data: https://ncovtracker.doh.gov.ph/ (click on box in top right labeled "Download COVID-19 COH Data Drop")*

Once in the google drive folder right-click file "DOH COVID Data Drop_ YYYYMMDD - 07 Testing Aggregates.csv" and copy the shareable link
e.g. https://drive.google.com/file/d/1GE-uO9kaFBgwreu7zFdXhYvG3U_9EY8C/view?usp=sharing
Update \autumun\inputs\covid_phl\fetch.py DATA URL = '1GE-uO9kaFBgwreu7zFdXhYvG3U_9EY8C'

2. Update scripts\phl_data_upload.py

From the same daily data link right-click file "DOH COVID Data Drop_ YYYYMMDD - 05 DOH Data Collect - Daily Report.csv" and copy the shareable link.
Update scripts\phl_data_upload.py e.g. PHL_doh_link=1WxoFhzZzglkk1RbOQAWI2gHeKkqwkD9P

For FASSSTER data use https://drive.google.com/drive/folders/1qnUsvq5SXxwdw9ttRtOojccVGHaYj6_k
Locate the latest ConfirmedCases_Final_YYYY-MM-DD.zip file, copy the shareable link and update
Update scripts\phl_data_upload.py e.g. PHL_fassster_link = "1sfwFryQP6lPutGxS62IIGUugDhRy_1h8"

The functions in phl_data_upload.py do the following:
1. Downloads data
2. Formats and filters data by region (using a duplicated dataset for the national model)
3. Calculated daily ICU occupancy, confirmed cases, and cumulative deaths
4. Dumps calibration data (step 3) into json files
5. Deletes files 

Run scripts\phl_data_upload.py in order to update.

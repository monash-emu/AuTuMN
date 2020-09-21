This readme documents the DHHS data update process.

1. Download the CSV from DHHS's secure folder to your local  AuTuMN repo folder data/inputs.
2. Rename the file to 'monashmodelextract.secret.csv'
3. Download the CSV report from CHRIS website to your local  AuTuMN repo folder data/inputs.
4. Rename the file to 'monitoringreport.secret.csv'
3. Run the dhhs_data_upload.py script located at \scripts

This will update a target.secret.json file for each region and generate a targets.encrypted.json file. 
The files are located in two folders, data/inputs/imports and apps/regions/<\region name>.

Select the all the '<>.encrypted.json' file and  data/secret-hashes.json and push to GitHub.
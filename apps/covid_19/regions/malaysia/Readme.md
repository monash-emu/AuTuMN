Calibration data for Malaysia is obtained from WPRO COVID modelling shared folder.
https://drive.google.com/drive/u/2/folders/1C-OsJbjdhQmpmedBZ8XuM65ANBO22dXc

In this folder, there are two files; a google sheet and an excel file.

Google sheet is titled: `Total Death Covid19 in Malaysia_updated 30 August 2020`
Location:	https://docs.google.com/spreadsheets/d/1cQe1k7GQRKFzcfXXxdL7_pTNQUMpOAPX3PqhPsI4xt8/edit?usp=sharing
Use case:	The deaths per day is extracted from the above sheets column A and C.

Excel sheet is titled is: `New _import_ICU cases in Malaysia_updated 30 August 2020`
Location:	https://drive.google.com/file/d/1mnZcmj2jfmrap1ytyg_ErD1DZ7zDptyJ/view?usp=sharing
Use case:	For case notifications, we extract column A (Date) and column B (New Cases (A)).
			For ICU occupancy, we extract column A (Date) and column F (Total ICU Usage including ventilator usage (E)).
			For case importation, we extract column A (Date) and column B (Imported cases (B))
			
The target values are extracted and the date index is calculated on a local excel spreadsheet. 
These values are then entered into the targets.json file for this region.
Case importations values are entered into default.yml.

Testing data are extracted from the OWID database (on the recommendation of in-country staff).
There is one very high relative to the other values, which is removed in pre-processing and the overall
testing numbers look somewhat low, but generally improve the calibration, so are used.

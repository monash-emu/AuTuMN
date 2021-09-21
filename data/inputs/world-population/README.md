# World Population Prospects 2019

This data is official United Nations population estimates and projections. More info [here](https://population.un.org/wpp/). The data can be downloaded [here](https://population.un.org/wpp/Download/Standard/Population/)

### [WPP2019_F01_LOCATIONS](https://population.un.org/wpp/Download/Files/4_Metadata/WPP2019_F01_LOCATIONS.XLSX)

This file contains metadata which we need to map ISO3 country codes to numerical country codes, which are used elsewhere. We prefer to use the ISO3 country codes.

### [WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/1_Population/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx>)

> Quinquennial Population by Five-Year Age Groups - Both Sexes. De facto population as of 1 July of the year indicated classified by five-year age groups (0-4, 5-9, 10-14, ..., 95-99, 100+). Data are presented in thousands.

This sheet is used for country-wide population estimates. Population numbers are reported in the thousands.

### subregions

This CSV contains population data for subregions of countries, like states and cities, which we want to run simulations on.
The UN WPP2019 dataset does not contain these figures.

TODO: Document the provenance of the region data.

Data for DHHS health clusters obtained from
Excel file 32350DS0003_2020.xls from [ABS site](https://www.abs.gov.au/statistics/people/population/regional-population-age-and-sex/latest-release#data-download)
and combined with 'LGA to Cluster mapping dictionary with proportions.csv' (.\data\inputs\mobility) to arrive at the static values for sub-regions.
Manual calculation are done in 'Sheet 1'. 

Malaysia regional population stats
[www.data.gov.my/data](https://www.data.gov.my/data/dataset/ec71711f-9b1f-42cd-9229-0c3e1f0e1dbb/resource/423f8f8c-5b74-4b5b-9ba9-d67a5c80e22c/download/mid-year-population-estimates-by-age-group-sex-and-state-malaysia-2015-2019.csv)

Philippines sub region population numbers are provided by the FASSSTER team except for 
[Davao City population](https://www.citypopulation.de/en/philippines/mindanao/admin/davao_del_sur/112402__davao/)
[Davao Region population](https://www.citypopulation.de/en/philippines/admin/11__davao/)

Sri Lanka sub region population
[Western Province](https://www.citypopulation.de/en/srilanka/prov/admin/1__western/)

Bali sub region population
[Bali](https://www.citypopulation.de/php/indonesia-admin.php?adm1id=51)

Vietnam Ho Chi Minh City population
Details are in /data/inputs/covid_vnm/README.md

### [WPP2019_FERT_F03_CRUDE_BIRTH_RATE](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/2_Fertility/WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx>)

Used for crude birth rate

> Number of births over a given period divided by the person-years lived by the population over that period. It is expressed as average annual number of births per 1,000 population.

### [WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/3_Mortality/WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx>)

Used for mortality

> Number of deaths by five-year age groups. Data are presented in thousands.

### [WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/3_Mortality/WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx>)

Used for life expectancy

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

### [WPP2019_FERT_F03_CRUDE_BIRTH_RATE](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/2_Fertility/WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx>)

Used for crude birth rate

> Number of births over a given period divided by the person-years lived by the population over that period. It is expressed as average annual number of births per 1,000 population.

### [WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/3_Mortality/WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx>)

Used for mortality

> Number of deaths by five-year age groups. Data are presented in thousands.

### [WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES](<https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/EXCEL_FILES/3_Mortality/WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx>)

Used for life expectancy

# To access John Hopkins covid-19 github repository and collate their daily updates into a pandas data frame.
# This file creates 4 pandas DF and a list with a DF for each daily report.
# TODO - combine all sources in to one DF 

from datetime import datetime
from datetime import timedelta
import pandas as pd

# who_covid_19_situation_reports to pandas data frame
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/who_covid_19_situation_reports/who_covid_19_sit_rep_time_series/who_covid_19_sit_rep_time_series.csv'
who_sit_rep = pd.read_csv(url)

# csse_covid_19_time_series data (three files)
list_of_url = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
               'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
               'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv']

# hard coded for now
ts_confirmed = pd.read_csv(list_of_url[0])
ts_deaths = pd.read_csv(list_of_url[1])
ts_recovered = pd.read_csv(list_of_url[2]) 

ts_confirmed.to_csv('covid_confirmed.csv')
ts_deaths.to_csv('covid_deaths.csv')
ts_recovered.to_csv('covid_recovered.csv')
exit()
# daily reports url
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'

end = datetime.today().date()
start = datetime(2020,1,22).date()
step = timedelta(1)

list_of_filenames = []

# generate a list of url+date strings for file name download
while start < end:

    list_of_filenames.append(start.strftime('%m-%d-%Y'))
    start += step

# A list to store the daily reports 
daily_reports = []

for each in list_of_filenames:

     daily_reports.append(pd.read_csv(url+'/'+each+'.csv'))



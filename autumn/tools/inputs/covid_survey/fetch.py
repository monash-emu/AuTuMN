import requests
import json
import pandas as pd
import os

from autumn.settings import INPUT_DATA_PATH


COVID_SURVEY_PATH = os.path.join(INPUT_DATA_PATH, "covid_survey")
TODAY = pd.to_datetime("today").date().strftime("%Y%m%d")


countries = ["Australia", "Malaysia", "Myanmar", "Philippines", "Sri Lanka"]
indicators = ["mask","avoid_contact"]


def fetch_covid_survey_data():

    for country in countries:
        for indicator in indicators:
            if country == "Australia":
                region = "Victoria"
                API_URL = f"https://covidmap.umd.edu/api/resources?indicator={indicator}&type=daily&country={country}&region={region}&daterange=20210101-{TODAY}"
            elif country == "Philippines":
                region = "National Capital Region"
                API_URL = f"https://covidmap.umd.edu/api/resources?indicator={indicator}&type=daily&country={country}&region={region}&daterange=20210101-{TODAY}"
            else:
                API_URL = f"https://covidmap.umd.edu/api/resources?indicator={indicator}&type=daily&country={country}&daterange=20210101-{TODAY}"

            # request data from api
            response = requests.get(API_URL).text

            # convert json data to dic data for use!
            jsonData = json.loads(response)

            # convert to pandas dataframe
            df = pd.DataFrame.from_dict(jsonData["data"])
            file_name = f"{indicator}_{country}.csv"
            file_full_path = os.path.join(COVID_SURVEY_PATH, file_name)
            df.to_csv(file_full_path, index=False)

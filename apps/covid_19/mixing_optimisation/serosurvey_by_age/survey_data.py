import pandas as pd

# format:
serosurvey_data = {
    "belgium":
        [
            {
                "time_range": [],
                "measures": [
                    {"age_range": [0, 5], "central": 0., "ci": []}
                ]
            }
        ],
}


def read_belgium_data():
    survey_list = []
    periods = {
        1: [90, 96],
        2: [111, 117],
        3: [139, 146],
        4: [160, 165],
        5: [181, 186]
    }
    df = pd.read_csv('belgium_data.csv')
    for period_id in list(periods.keys()):
        survey = {"time_range": periods[period_id]}
        survey["measures"] = []
        mask = df['period'] == float(period_id)
        survey_df = df[mask]
        for index, row in survey_df.iterrows():
            survey["measures"].append(
                {"age_range": reformat_belgium_agegroup(row["age"]), "central": row["best"], "ci": [row["low"], row["high"]]}
            )
        survey_list.append(survey)
    return survey_list


def reformat_belgium_agegroup(string):
    if string == "90+":
        return [90]
    else:
        age_low = string.split(",")[0].split("[")[1]
        age_high = string.split(",")[1][:-1]
        return [age_low, age_high]


serosurvey_data_perc = {
    "belgium": read_belgium_data()
}

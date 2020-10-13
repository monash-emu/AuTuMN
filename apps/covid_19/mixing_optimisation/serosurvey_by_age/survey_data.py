import pandas as pd
import json

# format:
# serosurvey_data = {
#     "belgium":
#         [
#             {
#                 "time_range": [],
#                 "measures": [
#                     {"age_range": [0, 5], "central": 0., "ci": []}
#                 ]
#             }
#         ],
# }


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
        return [int(age_low), int(age_high)]


def read_uk_data():
    """
    From Ward et al. reporting on REACT2 study
    :return:
    """
    survey = [
        {
            "time_range": [172, 195],  # 20 June to 13 July 2020
            "measures": [
                {"age_range": [18, 25], "central": 7.9, "ci": [7.3, 8.5]},
                {"age_range": [25, 35], "central": 7.8, "ci": [7.4, 8.3]},
                {"age_range": [35, 45], "central": 6.1, "ci": [5.7, 6.6]},
                {"age_range": [45, 55], "central": 6.4, "ci": [6.0, 6.9]},
                {"age_range": [55, 65], "central": 5.9, "ci": [5.5, 6.4]},
                {"age_range": [65, 75], "central": 3.2, "ci": [2.8, 3.6]},
                {"age_range": [75], "central": 3.3, "ci": [2.9, 3.8]},
            ]
        }
    ]

    return survey


def read_spain_data():
    df = pd.read_csv('spain_data.csv')
    survey = {
        "time_range": [118, 132],  # 27Apr - 11May
        "measures": []
    }
    for index, row in df.iterrows():
        val = row["perc_immu"]
        measure = {
            "age_range": reformat_spain_agegroup(row["age"]),
            "central": float(val.split(" (")[0]),
            "ci": [
                float(val.split("&&")[0].split("(")[1]),
                float(val.split("&&")[1].split(")")[0])
            ]
        }
        survey["measures"].append(measure)

    return [survey]


def reformat_spain_agegroup(string):
    if string == "90":
        return [90]
    elif string == "0":
        return [0, 1]
    else:
        return [int(string.split("&&")[0]), int(string.split("&&")[1]) + 1]


serosurvey_data_perc = {
    "belgium": read_belgium_data(),
    "united-kingdom": read_uk_data(),
    "spain": read_spain_data(),
}

with open('serosurvey_data.json', 'w') as json_file:
    json.dump(serosurvey_data_perc, json_file)

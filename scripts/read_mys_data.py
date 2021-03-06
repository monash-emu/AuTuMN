import pandas as pd
import copy

file_path = "../data/xls/MYS_cases.csv"

data = pd.read_csv(file_path, sep=",")

_new_cases = {
    "times": list(data["time"]),
    "values": list(data["new_cases"]),
}

_imported_cases = {
    "times": list(data["time"][data["imported"].notnull()]),
    "values": list(data["imported"][data["imported"].notnull()]),
}

_hospital_prev = {
    "times": list(data["time"][data["hospital_prev"].notnull()]),
    "values": list(data["hospital_prev"][data["hospital_prev"].notnull()]),
}

_icu_prev = {
    "times": list(data["time"][data["icu_prev"].notnull()]),
    "values": list(data["icu_prev"][data["icu_prev"].notnull()]),
}


# Printed data:
new_cases = {
    "times": [
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
    ],
    "values": [
        7,
        14,
        5,
        28,
        10,
        6,
        18,
        12,
        20,
        9,
        45,
        35,
        190,
        125,
        120,
        117,
        110,
        130,
        153,
        123,
        212,
        106,
        172,
        235,
        130,
        159,
        150,
        156,
        140,
        142,
        208,
        217,
        150,
        179,
        131,
        170,
        156,
        109,
        118,
        184,
        153,
        134,
        170,
        85,
        110,
        69,
        54,
        84,
        36,
        57,
        50,
        71,
        88,
        51,
        38,
        40,
        31,
        94,
        57,
        69,
        105,
        122,
        55,
        30,
        45,
        39,
        68,
        54,
        67,
        70,
        16,
        37,
        40,
        36,
        17,
        22,
        47,
        37,
        31,
    ],
}
imported_cases = {
    "times": [
        94,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
    ],
    "values": [
        154.0,
        72.0,
        25.0,
        12.0,
        11.0,
        52.0,
        7.0,
        0.0,
        1.0,
        1.0,
        4.0,
        1.0,
        0.0,
        13.0,
        3.0,
        4.0,
        0.0,
        0.0,
        6.0,
        5.0,
        21.0,
        2.0,
        10.0,
    ],
}

imported_cases_2 = copy.deepcopy(new_cases)
for i, time in enumerate(imported_cases_2["times"]):
    if time < 120:
        imported_cases_2["values"][i] *= 0.21
    else:
        j = imported_cases["times"].index(time)
        imported_cases_2["values"][i] = imported_cases["values"][j]

# print("$$$$$$$$$$")
# print(imported_cases_2)

imported_cases_2 = {
    "times": [
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
    ],
    "values": [
        1.47,
        2.94,
        1.05,
        5.88,
        2.1,
        1.26,
        3.78,
        2.52,
        4.2,
        1.89,
        9.45,
        7.35,
        39.9,
        26.25,
        25.2,
        24.57,
        23.099999999999998,
        27.3,
        32.129999999999995,
        25.83,
        44.519999999999996,
        22.259999999999998,
        36.12,
        49.35,
        27.3,
        33.39,
        31.5,
        32.76,
        29.4,
        29.82,
        43.68,
        45.57,
        31.5,
        37.589999999999996,
        27.509999999999998,
        35.699999999999996,
        32.76,
        22.89,
        24.779999999999998,
        38.64,
        32.129999999999995,
        28.14,
        35.699999999999996,
        17.849999999999998,
        23.099999999999998,
        14.49,
        11.34,
        17.64,
        7.56,
        11.969999999999999,
        10.5,
        14.91,
        18.48,
        10.709999999999999,
        7.9799999999999995,
        8.4,
        6.51,
        72.0,
        25.0,
        12.0,
        11.0,
        52.0,
        7.0,
        0.0,
        1.0,
        1.0,
        4.0,
        1.0,
        0.0,
        13.0,
        3.0,
        4.0,
        0.0,
        0.0,
        6.0,
        5.0,
        21.0,
        2.0,
        10.0,
    ],
}


total_pop = 32364904.0
for i, data_dict in enumerate([_new_cases, _imported_cases, _hospital_prev, _icu_prev]):

    print("#############")
    print(data_dict["times"])
    print(data_dict["values"])
    print([[x] for x in data_dict["values"]])

import os
from datetime import datetime

import numpy as np
import pytest

from autumn.tools.db import Database
from autumn.tools.inputs import database as input_database
from autumn.tools.inputs import (
    get_country_mixing_matrix, get_crude_birth_rate, get_death_rates_by_agegroup, get_life_expectancy_by_agegroup,
    get_population_by_agegroup,
)
from autumn.tools.inputs.demography.queries import downsample_quantity, downsample_rate
from autumn.models.covid_19.mixing_matrix.macrodistancing import get_mobility_data, weight_mobility_data


@pytest.mark.github_only
def test_build_input_database(tmpdir, monkeypatch):
    """
    Ensure we can build the input database with nothing crashing
    """
    input_db_path = os.path.join(tmpdir, "inputs.db")
    monkeypatch.setattr(input_database, "INPUT_DB_PATH", input_db_path)
    assert not os.path.exists(input_db_path)
    input_database.build_input_database(rebuild=True)
    assert os.path.exists(input_db_path)
    db = Database(input_db_path)
    expected_tables = set(["countries", "population", "birth_rates", "deaths", "life_expectancy"])
    assert set(db.table_names()).intersection(expected_tables) == expected_tables


def test_get_mobility_data():
    google_mobility_locations = {
        "work":
            {"workplaces": 1.},
        "other_locations":
            {"retail_and_recreation": 0.25,
             "grocery_and_pharmacy": 0.25,
             "parks": 0.25,
             "transit_stations": 0.25},
        "home":
            {"residential": 1.},
    }
    base_date = datetime(2020, 1, 1, 0, 0, 0)
    mob_df, days = get_mobility_data("AUS", "Victoria", base_date)
    mob_values_df = weight_mobility_data(mob_df, google_mobility_locations).round(2)
    assert days[:10] == [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    assert mob_values_df["work"][:10].to_list() == [
        1.03,
        0.98,
        1.18,
        1.13,
        1.13,
        1.14,
        1.15,
        1.04,
        0.97,
        1.17,
    ]
    assert mob_values_df["other_locations"][:10].to_list() == [
        0.96,
        1.02,
        1.09,
        0.92,
        0.96,
        1.01,
        1.05,
        1.13,
        1.09,
        1.05,
    ]


def test_get_country_mixing_matrix():
    mixing_matrix = get_country_mixing_matrix("home", "AUS")
    eps = 1e-8 * np.ones(mixing_matrix.shape)
    assert ((mixing_matrix - AUS_ALL_LOCATIONS_MIXING_MATRIX) < eps).all()


def test_get_death_rates_by_agegroup():
    age_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    country_iso_code = "AUS"
    death_rates, years = get_death_rates_by_agegroup(age_breakpoints, country_iso_code)
    # FIXME: ADD DEATH RATES
    # assert death_rates == []


def test_get_life_expectancy_by_agegroup():
    age_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    country_iso_code = "AUS"
    life_expectancy, years = get_life_expectancy_by_agegroup(age_breakpoints, country_iso_code)


def test_get_crude_birth_rate():
    country_iso_code = "AUS"
    birth_rates, years = get_crude_birth_rate(country_iso_code)
    assert years == [
        1952.5,
        1957.5,
        1962.5,
        1967.5,
        1972.5,
        1977.5,
        1982.5,
        1987.5,
        1992.5,
        1997.5,
        2002.5,
        2007.5,
        2012.5,
        2017.5,
    ]
    assert birth_rates == [
        22.981,
        22.742,
        21.427,
        19.932,
        19.124,
        15.847,
        15.556,
        15.100,
        14.768,
        13.524,
        12.801,
        13.800,
        13.379,
        12.879,
    ]


def test_get_population_by_agegroup():
    age_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    country_iso_code = "AUS"
    population = get_population_by_agegroup(
        age_breakpoints, country_iso_code, region=None, year=2020
    )
    assert population == [
        3308972,
        3130480,
        3375453,
        3718346,
        3306061,
        3107734,
        2651187,
        1846377,
        1055274,
    ]


def test_get_population_by_agegroup__with_region():
    age_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    country_iso_code = "AUS"
    population = get_population_by_agegroup(
        age_breakpoints, country_iso_code, region="Victoria", year=2020
    )
    assert population == [816027, 768266, 1019387, 999073, 847833, 778843, 650779, 446567, 268029]


def test_downsample_rate__with_no_change():
    old_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    new_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    old_rates = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    expected_new_rates = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    new_rates = downsample_rate(old_rates, old_breakpoints, 10, new_breakpoints)
    assert round_list(new_rates) == round_list(expected_new_rates)


def test_downsample_rate__with_division():
    old_breakpoints = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    new_breakpoints = [0, 10, 20, 30]
    old_rates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_new_rates = [1.5, 3.5, 5.5, 8]
    new_rates = downsample_rate(old_rates, old_breakpoints, 10, new_breakpoints)
    assert round_list(new_rates) == round_list(expected_new_rates)


def test_downsample_quantity_rate__with_no_change():
    old_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    new_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    old_amounts = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    expected_new_amounts = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    new_amounts = downsample_quantity(old_amounts, old_breakpoints, new_breakpoints)
    assert round_list(new_amounts) == round_list(expected_new_amounts)


def test_downsample_quantity_rate__with_divisions():
    old_breakpoints = [0, 5, 10, 15, 20, 25]
    new_breakpoints = [0, 7, 14, 21]
    old_amounts = [1, 2, 3, 4, 5, 6]
    expected_new_amounts = [1 + 0.8, 1.2 + 2.4, 0.6 + 4 + 1, 4 + 6]
    new_amounts = downsample_quantity(old_amounts, old_breakpoints, new_breakpoints)
    assert round_list(new_amounts) == round_list(expected_new_amounts)


def test_downsample_quantity__with_big_end_bucket():
    old_breakpoints = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    new_breakpoints = [0, 25]
    old_amounts = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    expected_new_amounts = [2.5, 6.5]
    new_amounts = downsample_quantity(old_amounts, old_breakpoints, new_breakpoints)
    assert round_list(new_amounts) == round_list(expected_new_amounts)


def round_list(items):
    return [round(i, 2) for i in items]


AUS_ALL_LOCATIONS_MIXING_MATRIX = np.array(
    [
        [
            0.659867910795608,
            0.503965302372055,
            0.214772978035646,
            0.094510153059056,
            0.158074593079577,
            0.331193847635042,
            0.617289713854548,
            0.622990492808764,
            0.229866592095014,
            0.077708910918047,
            0.072123717135639,
            0.035872769397665,
            0.029785475779417,
            0.009919626860288,
            0.004242125055983,
            0.006161956350397,
        ],
        [
            0.314776879411496,
            0.89546001462198,
            0.412465791190826,
            0.128925532733157,
            0.039332998502762,
            0.149588016221369,
            0.431338236989521,
            0.624945097357415,
            0.476872935028523,
            0.162493461891297,
            0.049997970893924,
            0.029618997770383,
            0.016726342407213,
            0.009524495024264,
            0.004178192655833,
            0.004356267475068,
        ],
        [
            0.132821425298119,
            0.405073037746635,
            1.43388859406275,
            0.398806156390403,
            0.055354667771211,
            0.034249201154477,
            0.119908332999482,
            0.375304849625763,
            0.562357849037187,
            0.271331672898659,
            0.07557929220176,
            0.018716369063677,
            0.01407395144204,
            0.010441878807721,
            0.007697757485305,
            0.004157599106112,
        ],
        [
            0.061564242616668,
            0.116366264405157,
            0.450972698652086,
            1.19511326973645,
            0.189480604932965,
            0.041826518587004,
            0.025562692593451,
            0.158732060060729,
            0.359678147942347,
            0.464636026178651,
            0.211951099326908,
            0.047125598860014,
            0.019890149708701,
            0.011908087874695,
            0.004919142980182,
            0.003073667936149,
        ],
        [
            0.1410221175676,
            0.064641634746752,
            0.083547057193617,
            0.402242846691883,
            1.19219311434584,
            0.248001072104032,
            0.04667582181322,
            0.022215710536328,
            0.13973930536196,
            0.435902652417925,
            0.281063077973612,
            0.103922611585634,
            0.020110325778317,
            0.00402532350833,
            0.005358871802343,
            0.003697144015926,
        ],
        [
            0.42515574402386,
            0.137257822446548,
            0.043092151970879,
            0.083336485049927,
            0.277888421473973,
            1.08983124653991,
            0.227317778823295,
            0.041119266881215,
            0.02085971437773,
            0.076336562337682,
            0.208132757012212,
            0.106444707320677,
            0.043403615097124,
            0.01078760603794,
            0.001525697203471,
            0.006816801049851,
        ],
        [
            0.632563691550639,
            0.529787741723304,
            0.221806492632609,
            0.041730291430993,
            0.064194502903013,
            0.225838048050893,
            0.93633735121159,
            0.236536331699803,
            0.071505441900821,
            0.022377398300706,
            0.040783967371983,
            0.062238322339302,
            0.059047267139578,
            0.010582930330781,
            0.005422528249365,
            0.003593738365132,
        ],
        [
            0.533818394233163,
            0.763952053343123,
            0.564592013891645,
            0.208467310889942,
            0.031192310653973,
            0.040858668599736,
            0.164458585230081,
            0.947228026273632,
            0.174174665333529,
            0.042037500858414,
            0.025259553449327,
            0.01857086790461,
            0.032816621847325,
            0.019280514301168,
            0.009448039358911,
            0.002791958693004,
        ],
        [
            0.241485286535544,
            0.535246817744383,
            0.726603372189449,
            0.447729223282834,
            0.109945461659975,
            0.034539911626987,
            0.077090153134669,
            0.181137315725871,
            0.772082088682895,
            0.162170318766118,
            0.037187081825638,
            0.007443449480203,
            0.021397179282304,
            0.025120305484699,
            0.009872422448987,
            0.006675748644677,
        ],
        [
            0.122279221752283,
            0.276168815462039,
            0.460515310603205,
            0.616757658090957,
            0.323893657325603,
            0.077888146848332,
            0.03070906114634,
            0.080237920360456,
            0.164072673346052,
            0.67377439653488,
            0.140820605080268,
            0.027284744708558,
            0.011768383764441,
            0.009064089711807,
            0.008389520383545,
            0.012393858997509,
        ],
        [
            0.202734983517432,
            0.16897459806975,
            0.305291715543044,
            0.411281445545597,
            0.390544558192932,
            0.234836343025081,
            0.091047989733313,
            0.048609750331367,
            0.079533526234508,
            0.187474603638022,
            0.668687055566836,
            0.135915974793711,
            0.026984514709262,
            0.008731254168598,
            0.009396279234975,
            0.020105853715754,
        ],
        [
            0.327263174181627,
            0.324731467982267,
            0.218960740204972,
            0.302639598157039,
            0.283606111231349,
            0.328313321166328,
            0.257037801860177,
            0.084280807016141,
            0.042615286322583,
            0.123359665604138,
            0.219649901404088,
            0.70332540237903,
            0.147409416556862,
            0.043435343372585,
            0.007159855921403,
            0.020523984528288,
        ],
        [
            0.390012387335124,
            0.348613331036363,
            0.246459737180013,
            0.221557815493274,
            0.154865962920624,
            0.216622441983485,
            0.292573924524049,
            0.210710055129864,
            0.095468597908586,
            0.050066342694397,
            0.099139798037307,
            0.205579710713488,
            0.673060927337745,
            0.117825410579146,
            0.022641177935706,
            0.005231840566219,
        ],
        [
            0.276912769348463,
            0.408506520005129,
            0.36937363092212,
            0.194775379097249,
            0.133838509701755,
            0.154876998293871,
            0.265935324365622,
            0.315638661928782,
            0.260623596143589,
            0.081602943008362,
            0.079351975957155,
            0.113753332732554,
            0.165434121232626,
            0.679889417407158,
            0.102948555503362,
            0.014577679093468,
        ],
        [
            0.124528364220975,
            0.37307572376716,
            0.331749202445228,
            0.264715200347894,
            0.053835490535729,
            0.113357660616225,
            0.102258130285998,
            0.245019174433193,
            0.234382140447737,
            0.184045835332283,
            0.109950863051364,
            0.053180133489189,
            0.104953437033771,
            0.157145996090452,
            0.434751051004558,
            0.096379251405798,
        ],
        [
            0.223935664795901,
            0.293988435583329,
            0.46812652205773,
            0.366285350077597,
            0.094773166933007,
            0.08384270344864,
            0.104392568632524,
            0.23522078879905,
            0.287783012466083,
            0.263886336144683,
            0.296622611540382,
            0.126777547730481,
            0.049124115726431,
            0.09415959885167,
            0.100210364737509,
            0.304142982924456,
        ],
    ]
)

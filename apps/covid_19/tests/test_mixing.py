from apps.covid_19.preprocess.mixing_matrix import build_dynamic


def test_dynamic_mixing():
    google_mobility_locations = {
        "work": ["workplaces"],
        "other_locations": [
            "retail_and_recreation",
            "grocery_and_pharmacy",
            "parks",
            "transit_stations",
        ],
    }
    mixing_params = {
        "other_locations": {
            "append": True,
            "times": [220, 230, 240, 250, 260],
            "values": [1, 0.4, 0.3, 0.3, 0.5],
        },
        "work": {
            "append": True,
            "times": [220, 230, 240, 250, 260],
            "values": [1, 0.9, 0.5, 0.3, 0.6],
        },
        "school": {"append": False, "times": [46, 220], "values": [1, 0],},
    }
    mm_func = build_dynamic(
        country_iso3="MYS",
        region=None,
        mixing=mixing_params,
        npi_effectiveness_params={},
        google_mobility_locations=google_mobility_locations,
        is_periodic_intervention=False,
        periodic_int_params={},
        end_time=365,
        microdistancing_params={}
    )
    mm = mm_func(50)
    assert mm.shape == (16, 16)

from apps.covid_19.preprocess.mixing_matrix import build_dynamic


def test_dynamic_mixing():
    mixing_params = {
        "other_locations_times": [50, 60, 80, 100, 120],
        "other_locations_values": [1, 0.4, 0.3, 0.3, 0.5],
        "work_times": [50, 60, 80, 100, 120],
        "work_values": [1, 0.9, 0.5, 0.3, 0.6],
        "school_times": ["20200312", "20200314"],
        "school_values": [1, 0],
    }
    mm_func = build_dynamic(
        country="Malaysia",
        mixing_params=mixing_params,
        npi_effectiveness_params={},
        is_reinstate_regular_prayers=False,
        prayers_params={},
        end_time=365,
    )
    mm = mm_func(50)
    assert mm.shape == (16, 16)

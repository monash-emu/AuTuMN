import numpy as np
from numpy.testing import assert_allclose

from autumn.models.covid_19.parameters import TimeSeries
from autumn.models.sm_sir.mixing_matrix.mixing_adjusters import AgeMixingAdjuster


def test_age_adjuster__with_empty_mixing_data():
    age_mixing = {}
    adjuster = AgeMixingAdjuster(age_mixing)
    assert adjuster.adjustment_funcs == {}
    mm = np.ones([16, 16])
    adj_mm = adjuster.get_adjustment(0, mm)
    assert_allclose(mm, adj_mm, atol=0.01, verbose=True)


def test_age_adjuster__with_mixing_data():
    age_mixing = {
        "0": TimeSeries(
            times=[0, 1, 2, 3],
            values=[2, 2, 2, 2],
        ),
        "5": TimeSeries(
            times=[0, 1, 2, 3],
            values=[3, 3, 3, 3],
        ),
    }
    adjuster = AgeMixingAdjuster(age_mixing)
    assert set(adjuster.adjustment_funcs.keys()) == {"0", "5"}
    input_mm = np.ones([16, 16])
    adj_mm = adjuster.get_adjustment(0, input_mm)
    expected_mm = np.ones([16, 16])
    # Add adjustment of 2 to the 1st row and col for the 0-5 age bracket
    expected_mm[0, :] *= 2
    expected_mm[:, 0] *= 2
    # Add adjustment of 3 to the 2nd row and col for the 5-10 age bracket
    expected_mm[1, :] *= 3
    expected_mm[:, 1] *= 3
    assert_allclose(expected_mm, adj_mm, atol=0.01, verbose=True)

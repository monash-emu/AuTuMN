import numpy as np

from autumn.tools.utils.utils import get_apply_odds_ratio_to_prop


def test_apply_odds_ratio_to_prop(n_props_to_test=100, n_ratios_to_test=100, error=1e-6):
    """
    Test that the results makes sense when we apply an odds ratio to a proportion.
    """

    # Get some raw proportions and some odds ratios to apply
    proportions = np.random.uniform(size=n_props_to_test)
    odds_ratios = np.random.uniform(size=n_ratios_to_test)

    for i_prop in proportions:
        for i_ratio in odds_ratios:

            # Use the function
            or_to_prop_func = get_apply_odds_ratio_to_prop(i_ratio)
            result = or_to_prop_func(i_prop)

            # Make sure the result is still a proportion
            assert 0.0 <= result <= 1.0

            # Make sure it goes in the same direction as the odds ratio request
            if i_ratio > 1.0:
                assert result > i_prop
            elif i_ratio == 1.0:
                assert abs(result - i_prop) < error
            elif i_ratio < 1.0:
                assert result < i_prop

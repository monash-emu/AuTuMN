import numpy as np

from autumn.tool_kit.utils import apply_odds_ratio_to_proportion


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
            result = apply_odds_ratio_to_proportion(i_prop, i_ratio)

            # Make sure the result is a proportion
            assert 0. <= result <= 1.

            # Make sure it goes in the same direction as the odds ratio request
            if i_ratio > 1.:
                assert result > i_prop
            elif i_ratio == 1.:
                assert abs(result - i_prop) < error
            elif i_ratio < 1.:
                assert result < i_prop

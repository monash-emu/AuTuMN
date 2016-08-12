
import numpy

def discounting_fx (x_vals, cost_uninflated, discount_rate, current_year_pos):

        cost_todiscount = cost_uninflated
        if x_vals <= 2015:
            years_into_future = 0
        elif x_vals > 2015 and x_vals <= 2016:
            years_into_future = 1
        elif x_vals > 2016 and x_vals <= 2017:
            years_into_future = 2
        elif x_vals > 2017 and x_vals <= 2018:
            years_into_future = 3
        elif x_vals > 2018 and x_vals <= 2019:
            years_into_future = 4
        elif x_vals > 2019 and x_vals <= 2020:
            years_into_future = 5
        elif x_vals > 2020 and x_vals <= 2021:
            years_into_future = 6
        elif x_vals > 2021 and x_vals <= 2022:
            years_into_future = 7
        elif x_vals > 2022 and x_vals <= 2023:
            years_into_future = 8
        elif x_vals > 2023 and x_vals <= 2024:
            years_into_future = 9
        elif x_vals > 2024 and x_vals <= 2025:
            years_into_future = 10
        elif x_vals > 2025 and x_vals <= 2026:
            years_into_future = 11
        elif x_vals > 2026 and x_vals <= 2027:
            years_into_future = 12
        elif x_vals > 2027 and x_vals <= 2028:
            years_into_future = 13
        elif x_vals > 2028 and x_vals <= 2029:
            years_into_future = 14
        elif x_vals > 2029 and x_vals <= 2030:
            years_into_future = 15
        elif x_vals > 2030 and x_vals <= 2031:
            years_into_future = 16
        elif x_vals > 2031 and x_vals <= 2032:
            years_into_future = 17
        elif x_vals > 2032 and x_vals <= 2033:
            years_into_future = 18
        elif x_vals > 2033 and x_vals <= 2034:
            years_into_future = 19
        else:
            years_into_future = 20
        cost_discounted = cost_todiscount / ((1 + discount_rate)**years_into_future)

        return cost_discounted
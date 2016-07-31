
import numpy
import matplotlib.pyplot as plt
import math
import scipy.optimize

saturation = 0.7
unit_cost = 15.
popsize = 1.e3
c_reflection_cost = 20000
alpha = 1.
init_cost = 1.e4

start_coverage = 0.0001
end_coverage = saturation
delta_coverage = 0.001

######## MAKE COVERAGE RANGE #######################

def make_coverage_steps(start_coverage, end_coverage, delta_coverage):
    steps = []
    step = start_coverage
    while step <= end_coverage:
        steps.append(step)
        step += delta_coverage
    return steps
coverage_values = make_coverage_steps(start_coverage, end_coverage, delta_coverage)

c_reflection_cost_equal_initcost = False


if c_reflection_cost_equal_initcost is True:
    a = saturation / (1 - 2**alpha)
    b_growth_rate = (2**(alpha + 1)) / (alpha * (saturation - a) * unit_cost * popsize)

else:
#     def F(b):
#         a = saturation - (2.**(alpha + 1.)) / (alpha * b * unit_cost * popsize)
#         print(a)
#         result = (b - numpy.log((abs((a - saturation)/a)**(1./alpha)) -1.) / (c_reflection_cost - init_cost))
#         print(result)
#         return result
#
#     b_max = (2.**(alpha+1.)) / (alpha*saturation*unit_cost*popsize)
#     print(F(b_max))
#
#     b = scipy.optimize.brentq(F,1e-5,b_max*0.999)
#
# #    b = scipy.optimize.broyden1(F, 1.e-10, f_tol = 1e-14)
#     b_growth_rate = b
#     print('b_growth_rate ' + str(b_growth_rate))
#     a = (saturation - 2**(alpha + 1) / (alpha * b * unit_cost * popsize))
#     print('a ' + str(a))

    def F(a):
        b = (2**(alpha + 1)) / (alpha * (saturation - a) * unit_cost * popsize)

        result = a - saturation/(1-(1 + math.exp(-b*(init_cost-c_reflection_cost)))**alpha)
        print(result)
        return(result)
    a = scipy.optimize.broyden1(F, 0., f_tol = 1e-25)
    b_growth_rate = (2**(alpha + 1)) / (alpha * (saturation - a) * unit_cost * popsize)


def get_cost_from_coverage(saturation, coverage, c_reflection_cost, alpha, b_growth_rate, a):
    cost = c_reflection_cost - 1/b_growth_rate * math.log((((saturation - a) / (coverage - a))**(1 / alpha)) - 1)
    return cost

cost = []

for coverage in coverage_values:
    cost.append(get_cost_from_coverage(saturation,
                                    coverage,
                                    c_reflection_cost,
                                    alpha,
                                    b_growth_rate,
                                    a))

plt.figure('Cost-coverage curve')
plt.plot(cost, coverage_values, 'r', linewidth = 3)
plt.ylim([0, 1])
plt.show()


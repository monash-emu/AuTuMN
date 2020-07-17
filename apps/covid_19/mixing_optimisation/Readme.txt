The configurations will be all combinations of the following options:
- 6 countries
  belgium, united-kingdom, spain, italy, sweden, france.

- 2 durations for the phase during which herd immunity is being built:
  6 months and 12 months  (config=2, config=3)

- 2 objectives to minimise
  Deaths and Years of life lost, minimised separately (mono-objective optimisation). 

- 2 optimisation modes
   - “by age”: social mixing is mitigated according to age
   - “by location”: social mixing is mitigated by location type (schools, workplaces and locations other than schools, workplaces and homes)

- Multiple uncertainty parameter samples
  Each optimisation scenario will be run multiple times using N different calibrated parameter sets (MCMC posterior samples) to account for uncertainty. We'll probably use N=100.

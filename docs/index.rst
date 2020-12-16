Summer documentation
==================================

Summer is a compartmental disease modelling framework, written in Python.
It provides a high-level API to build and run models. Features include:

- A variety of inter-compartmental flows (infections, sojourn, fractional, births, deaths, imports)
- Force of infection multipliers (frequency, density)
- Post-processing of compartment sizes into derived outputs
- Stratification of compartments, including:
   - Adjustments to flow rates based on strata
   - Adjustments to infectiousness based on strata
   - Heterogeneous mixing between strata
   - Multiple disease strains

Contents
--------

.. toctree::
   :maxdepth: 2

   examples/index.rst
   api/index.rst
   
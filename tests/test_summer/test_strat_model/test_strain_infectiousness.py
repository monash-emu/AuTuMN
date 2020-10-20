"""
Tests for the 'strains' feature of SUMMER.

Strains allows for multiple concurrent infections, which have different properties.

- all infected compartments are stratified into strains (all, not just diseased or infectious, etc)
- assume that a person can only have one strain (simplifying assumption)
- strains can have different infectiousness, mortality rates, etc (set via flow adjustment)
- strains can progress from one to another (inter-strain flows)
- each strain has a different force of infection calculation
- any strain stratification you must be applied to all infected compartments

Force of infection:

- we have multiple infectious populations (one for each strain)
- people infected by a particular strain get that strain

"""


def test_strains__with_one_not_infectious():
    """
    If there are two strains, one which is not infectious, and another which is infectious.
    Then we expect the force of infection to ???
    """
    pass


def test_strains__with_two_symmetric_strains():
    """
    Adding two strains with the same properties should yield the same infection dynamics
    and outputs as having no strains at all.
    We expect the force of infection for each strain to be ??? of the unstratified model.
    """
    pass

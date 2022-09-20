import os
import warnings

# Ignore future warnings they're annoying.
warnings.simplefilter(action="ignore", category=FutureWarning)

# Ensure NumPy only uses 1 thread for matrix multiplication,
# because NumPy is stupid and tries to use heaps of threads,
#  which is quite wasteful and makes our models run way more slowly.
# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
os.environ["OMP_NUM_THREADS"] = "1"

from autumn.core.registry import register_project
from autumn.settings import Models, Region

# TB projects
register_project(
    Models.TB,
    Region.MARSHALL_ISLANDS,
    "autumn.projects.tuberculosis.marshall_islands.project",
)
register_project(Models.TB, Region.PHILIPPINES, "autumn.projects.tuberculosis.philippines.project")

# Example projects
register_project(
    Models.EXAMPLE,
    Region.PHILIPPINES,
    "autumn.projects.example.philippines.project",
)

# COVID: European mixing optimization
register_project(
    Models.COVID_19,
    Region.BELGIUM,
    "autumn.projects.covid_19.mixing_optimisation.regions.belgium.project",
)
register_project(
    Models.COVID_19,
    Region.SPAIN,
    "autumn.projects.covid_19.mixing_optimisation.regions.spain.project",
)
register_project(
    Models.COVID_19,
    Region.SWEDEN,
    "autumn.projects.covid_19.mixing_optimisation.regions.sweden.project",
)
register_project(
    Models.COVID_19,
    Region.UNITED_KINGDOM,
    "autumn.projects.covid_19.mixing_optimisation.regions.united_kingdom.project",
)
register_project(
    Models.COVID_19,
    Region.ITALY,
    "autumn.projects.covid_19.mixing_optimisation.regions.italy.project",
)
register_project(
    Models.COVID_19,
    Region.FRANCE,
    "autumn.projects.covid_19.mixing_optimisation.regions.france.project",
)

# Sri Lanka
register_project(
    Models.COVID_19,
    Region.SRI_LANKA,
    "autumn.projects.covid_19.sri_lanka.sri_lanka.project",
)

register_project(
    Models.SM_SIR,
    Region.NCR,
    "autumn.projects.sm_sir.philippines.national-capital-region.project",
)


register_project(
    Models.SM_SIR,
    Region.BARMM,
    "autumn.projects.sm_sir.philippines.barmm.project",
)


register_project(
    Models.SM_SIR,
    Region.WESTERN_VISAYAS,
    "autumn.projects.sm_sir.philippines.western-visayas.project",
)


register_project(
    Models.SM_SIR,
    Region.MALAYSIA,
    "autumn.projects.sm_sir.malaysia.malaysia.project",
)
register_project(
    Models.SM_SIR,
    Region.MYANMAR,
    "autumn.projects.sm_sir.myanmar.myanmar.project",
)
register_project(
    Models.SM_SIR,
    Region.BANGLADESH,
    "autumn.projects.sm_sir.bangladesh.bangladesh.project",
)
register_project(
    Models.SM_SIR,
    Region.DHAKA,
    "autumn.projects.sm_sir.bangladesh.dhaka.project",
)
register_project(
    Models.SM_SIR,
    Region.COXS_BAZAR,
    "autumn.projects.sm_sir.bangladesh.coxs_bazar.project",
)
register_project(
    Models.SM_SIR,
    Region.HO_CHI_MINH_CITY,
    "autumn.projects.sm_sir.vietnam.ho_chi_minh_city.project",
)

register_project(
    Models.SM_SIR,
    Region.HANOI,
    "autumn.projects.sm_sir.vietnam.hanoi.project",
)
register_project(
    Models.SM_SIR,
    Region.BHUTAN,
    "autumn.projects.sm_sir.bhutan.bhutan.project",
)

register_project(
    Models.SM_SIR,
    Region.THIMPHU,
    "autumn.projects.sm_sir.bhutan.thimphu.project",
)


register_project(
    Models.SM_SIR,
    Region.NORTHERN_TERRITORY,
    "autumn.projects.sm_sir.australia.northern_territory.project",
)


# Hierarchical SIR project
register_project(
    Models.HIERARCHICAL_SIR,
    Region.MULTI,
    "autumn.projects.hierarchical_sir.multi.project",
)

# New TB Dynamics
register_project(
    Models.TBD,
    Region.KIRIBATI,
    "autumn.projects.tb_dynamics.kiribati.project",
)

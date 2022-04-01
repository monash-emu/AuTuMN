import os
import warnings

# Ignore future warnings they're annoying.
warnings.simplefilter(action="ignore", category=FutureWarning)

# Ensure NumPy only uses 1 thread for matrix multiplication,
# because NumPy is stupid and tries to use heaps of threads,
#  which is quite wasteful and makes our models run way more slowly.
# https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
os.environ["OMP_NUM_THREADS"] = "1"


from autumn.settings import Models, Region
from autumn.tools.registry import register_project


# TB projects
register_project(
    Models.TB, Region.MARSHALL_ISLANDS, "autumn.projects.tuberculosis.marshall_islands.project"
)
register_project(Models.TB, Region.PHILIPPINES, "autumn.projects.tuberculosis.philippines.project")

# Example projects
register_project(Models.EXAMPLE, Region.PHILIPPINES, "autumn.projects.example.philippines.project")


register_project(
    Models.COVID_19,
    Region.NORTH_EAST_METRO,
    "autumn.projects.covid_19.victoria.north_east_metro.project",
)

register_project(
    Models.COVID_19,
    Region.SOUTH_EAST_METRO,
    "autumn.projects.covid_19.victoria.south_east_metro.project",
)

register_project(
    Models.COVID_19,
    Region.WEST_METRO,
    "autumn.projects.covid_19.victoria.west_metro.project",
)

register_project(
    Models.COVID_19,
    Region.BARWON_SOUTH_WEST,
    "autumn.projects.covid_19.victoria.barwon_south_west.project",
)

register_project(
    Models.COVID_19,
    Region.GIPPSLAND,
    "autumn.projects.covid_19.victoria.gippsland.project",
)

register_project(
    Models.COVID_19,
    Region.GRAMPIANS,
    "autumn.projects.covid_19.victoria.grampians.project",
)

register_project(
    Models.COVID_19,
    Region.HUME,
    "autumn.projects.covid_19.victoria.hume.project",
)

register_project(
    Models.COVID_19,
    Region.LODDON_MALLEE,
    "autumn.projects.covid_19.victoria.loddon_mallee.project",
)

register_project(
    Models.COVID_19,
    Region.VICTORIA,
    "autumn.projects.covid_19.victoria.victoria.project",
)

# COVID: European mixing optmization
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

# COVID: Philippines project
register_project(
    Models.COVID_19, Region.CALABARZON, "autumn.projects.covid_19.philippines.calabarzon.project"
)
register_project(
    Models.COVID_19,
    Region.CENTRAL_VISAYAS,
    "autumn.projects.covid_19.philippines.central_visayas.project",
)

register_project(
    Models.COVID_19, Region.DAVAO_CITY, "autumn.projects.covid_19.philippines.davao_city.project"
)

register_project(
    Models.COVID_19,
    Region.DAVAO_REGION,
    "autumn.projects.covid_19.philippines.davao_region.project",
)

register_project(
    Models.COVID_19, Region.MANILA, "autumn.projects.covid_19.philippines.manila.project"
)
register_project(
    Models.COVID_19, Region.PHILIPPINES, "autumn.projects.covid_19.philippines.philippines.project"
)


# COVID: Malaysia project
register_project(Models.COVID_19, Region.JOHOR, "autumn.projects.covid_19.malaysia.johor.project")
register_project(
    Models.COVID_19, Region.KUALA_LUMPUR, "autumn.projects.covid_19.malaysia.kuala_lumpur.project"
)
register_project(
    Models.COVID_19, Region.MALAYSIA, "autumn.projects.covid_19.malaysia.malaysia.project"
)

register_project(Models.COVID_19, Region.PENANG, "autumn.projects.covid_19.malaysia.penang.project")
register_project(Models.COVID_19, Region.SABAH, "autumn.projects.covid_19.malaysia.sabah.project")
register_project(
    Models.COVID_19, Region.SELANGOR, "autumn.projects.covid_19.malaysia.selangor.project"
)

# Sri Lanka
register_project(
    Models.COVID_19, Region.SRI_LANKA, "autumn.projects.covid_19.sri_lanka.sri_lanka.project"
)

# Vietnam
register_project(
    Models.COVID_19, Region.VIETNAM, "autumn.projects.covid_19.vietnam.vietnam.project"
)
register_project(
    Models.COVID_19,
    Region.HO_CHI_MINH_CITY,
    "autumn.projects.covid_19.vietnam.ho_chi_minh_city.project",
)

register_project(
    Models.COVID_19,
    Region.HANOI,
    "autumn.projects.covid_19.vietnam.hanoi.project",
)


register_project(Models.COVID_19, Region.MYANMAR, "autumn.projects.covid_19.myanmar.project")

# COVID: Victoria project
# FIXME: Parameter validation issues
# register_project(Models.COVID_19, Region.VICTORIA, "autumn.projects.covid_19.victoria.project")


register_project(
    Models.SM_SIR, Region.NCR, "autumn.projects.sm_sir.philippines.national-capital-region.project"
)
register_project(Models.SM_SIR, Region.MALAYSIA, "autumn.projects.sm_sir.malaysia.malaysia.project")
register_project(Models.SM_SIR, Region.MYANMAR, "autumn.projects.sm_sir.myanmar.myanmar.project")
register_project(
    Models.SM_SIR, Region.BANGLADESH, "autumn.projects.sm_sir.bangladesh.bangladesh.project"
)
register_project(Models.SM_SIR, Region.DHAKA, "autumn.projects.sm_sir.bangladesh.dhaka.project")
register_project(
    Models.SM_SIR, Region.COXS_BAZAR, "autumn.projects.sm_sir.bangladesh.coxs_bazar.project"
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

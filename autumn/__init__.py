import warnings

# Ignore future warnings they're annoying.
warnings.simplefilter(action="ignore", category=FutureWarning)

from autumn.settings import Models, Region
from autumn.tools.project import register_project


# TB projects
register_project(
    Models.TB, Region.MARSHALL_ISLANDS, "autumn.projects.tuberculosis.marshall_islands.project"
)
register_project(Models.TB, Region.PHILIPPINES, "autumn.projects.tuberculosis.philippines.project")

# Example projects
register_project(Models.EXAMPLE, Region.PHILIPPINES, "autumn.projects.example.philippines.project")
register_project(Models.EXAMPLE, Region.VICTORIA, "autumn.projects.example.victoria.project")

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

# Nepal
register_project(Models.COVID_19, Region.NEPAL, "autumn.projects.covid_19.nepal.project")

# Sri Lanka
register_project(Models.COVID_19, Region.SRI_LANKA, "autumn.projects.covid_19.sri_lanka.project")

# COVID: Victoria project
# FIXME: Parameter validation issues
# register_project(Models.COVID_19, Region.VICTORIA, "autumn.projects.covid_19.victoria.project")

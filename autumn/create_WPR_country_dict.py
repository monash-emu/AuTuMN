import json

WPR_countries = {
    "region": ["malaysia", "australia"],

    "path": ["autumn.projects.sm_sir.WPRO.malaysia.malaysia.project",
             "autumn.projects.sm_sir.WPRO.australia.australia.project"]
}

with open("wpro_list.json", "w") as outfile:
    json.dump(WPR_countries, outfile)
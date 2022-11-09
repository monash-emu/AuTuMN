import json

WPR_countries = {
    "region": ["malaysia",
               "australia",
               "philippines",
               "vietnam"

],

    "path": ["autumn.projects.sm_sir.WPRO.malaysia.malaysia.project",
             "autumn.projects.sm_sir.WPRO.australia.australia.project",
             "autumn.projects.sm_sir.WPRO.philippines.philippines.project",
             "autumn.projects.sm_sir.WPRO.vietnam.vietnam.project",
             ]
}

with open("wpro_list.json", "w") as outfile:
    json.dump(WPR_countries, outfile)
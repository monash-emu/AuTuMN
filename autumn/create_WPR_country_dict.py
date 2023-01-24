import json

WPR_countries = {
    "region": ["malaysia",
               "australia",
               "philippines",
               "vietnam",
               "japan",
               "mongolia",
               "new-zealand",
               "south-korea",
               "singapore",
               ],

    "path": ["autumn.projects.sm_sir.WPRO.malaysia.malaysia.project",
             "autumn.projects.sm_sir.WPRO.australia.australia.project",
             "autumn.projects.sm_sir.WPRO.philippines.philippines.project",
             "autumn.projects.sm_sir.WPRO.vietnam.vietnam.project",
             "autumn.projects.sm_sir.WPRO.japan.japan.project",
             "autumn.projects.sm_sir.WPRO.mongolia.mongolia.project",
             "autumn.projects.sm_sir.WPRO.new-zealand.new-zealand.project",
             "autumn.projects.sm_sir.WPRO.south-korea.south-korea.project",
             "autumn.projects.sm_sir.WPRO.singapore.singapore.project",
              ]
}

with open("wpro_list.json", "w") as outfile:
    json.dump(WPR_countries, outfile)
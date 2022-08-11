from autumn.projects.sm_sir.WPRO.WPR_constants import WPR_Countries
from autumn.core.project import build_rel_path
import re

for country in WPR_Countries:
    register_folder_path = build_rel_path(f"register_WPRO.py")
    register_folder_path = f"{register_folder_path}"

    init_file_path = build_rel_path(f"__init__.py")
    init_file_path = f"{init_file_path}"

    # registering each country by changing country name in register_WPRO
    with open(register_folder_path, 'r+') as register_file:
        read_file = register_file.read()
        register_file.seek(0)

        country_text = country
        if '-' in country_text:
            country_text = country_text.replace('-', '_')
        region_name = country_text.upper()  # get the corresponding iso3 code

        read_file = re.sub("COUNTRY", f"{region_name}", read_file)
        read_file = re.sub("country", f"{country}", read_file)

        with open(init_file_path, 'a+') as init_file:
            init_file.write(read_file)
            init_file.truncate()


import os
import json

from autumn.tools.project.params import read_param_value_from_string
from autumn.settings import MODELS_PATH, DOCS_PATH


def write_params_to_tex(project, params_list, project_path, output_dir_path=None):
    """
    Write a parameter table as a tex file
    :param project: Project object
    :param params_list: ordered list of parameters to be included
    :param project_path: path of the project's directory
    :param output_dir_path: path of the directory where to dump the output tex file.
           Default is "docs/papers/<model_name>/projects/<region_name>".
    """
    # Load parameters' descriptions (base model)
    base_params_descriptions_path = os.path.join(MODELS_PATH, project.model_name, "params_descriptions.json")
    with open(base_params_descriptions_path, mode="r") as f:
        params_descriptions = json.load(f)

    # Load parameters' descriptions (project-specific)
    updated_descriptions_path = os.path.join(project_path, "params_descriptions.json")
    if os.path.isfile(updated_descriptions_path):
        with open(updated_descriptions_path, mode="r") as f:
            updated_params_descriptions = json.load(f)
        params_descriptions.update(updated_params_descriptions)

    # Write to tex file
    if output_dir_path is None:
        output_dir_path = os.path.join(DOCS_PATH, "papers", project.model_name, "projects", project.region_name)

    tex_file_path = os.path.join(output_dir_path, "parameters_auto.tex")
    with open(tex_file_path, "w") as tex_file:
        write_param_table_header(tex_file)
        # Write one row per parameter
        for param in params_list:
            param_info = params_descriptions[param]
            
            # read raw parameter value
            value = read_param_value_from_string(project.param_set.baseline.to_dict(), param)

            # apply potential multiplier
            multiplier = 1 if "multiplier" not in param_info else param_info["multiplier"]
            display_value = multiplier * value

            # round up the value
            if "n_digits" in param_info:
                display_value = round(display_value, param_info['n_digits'])

            # convert to string
            display_value = str(display_value)

            # add potential unit
            if "unit" in param_info:
                display_value += f" {param_info['unit']}"

            # write table row
            table_row = f"\\hline {param_info['full_name']} & {display_value} & "
            if "rationale" in param_info:
                table_row += param_info["rationale"]
            tex_file.write("\t " + table_row + " \n")

        # Finish up the table
        tex_file.write("\\end{longtable}")


def write_param_table_header(tex_file):
    # Write the table header
    tex_file.write("\\begin{longtable}[ht]{| >{\\raggedright}p{4cm} | >{\\raggedright}p{3cm} | p{6.8cm} |} \n")
    tex_file.write("\t \hline \n")
    tex_file.write("\t Parameter & Value & Rationale \\\ \n")
    tex_file.write("\t \endfirsthead \n")
    tex_file.write("\t \\multicolumn{3}{c}{continuation of parameters table} \\\ \n ")
    tex_file.write("\t \endhead \n \n")

import os

from autumn.core.project.params import read_param_value_from_string

DISTRIBUTION_TRANSLATE = {
    "trunc_normal": "truncated normal",
}


def write_main_param_table(
    project,
    main_table_params_list,
    params_descriptions,
    all_calibration_params_names,
    all_priors,
    output_dir_path,
):
    tex_file_path = os.path.join(output_dir_path, "parameters_auto.tex")
    with open(tex_file_path, "w") as tex_file:
        write_param_table_header(tex_file, table_type="main_table")
        # Write one row per parameter
        for param in main_table_params_list:
            if param not in params_descriptions:
                print(
                    f"Warning: parameter {param} won't be in the main table as there is no description available"
                )
                continue

            param_info = params_descriptions[param]

            if param in all_calibration_params_names:
                display_value = get_prior_display(
                    all_priors[all_calibration_params_names.index(param)],
                    table_type="main_table",
                )
            else:
                display_value = get_fixed_param_display(project, param, param_info)

            # write table row
            table_row = f"\\hline {param_info['full_name']} & {display_value} & "
            if "rationale" in param_info:
                table_row += param_info["rationale"]
            tex_file.write("\t " + table_row + " \n")

        # Finish up the table
        tex_file.write("\n \\hline \n \\end{longtable}")


def write_priors_table(params_descriptions, all_priors, output_dir_path):
    tex_file_path = os.path.join(output_dir_path, "parameters_priors_auto.tex")
    with open(tex_file_path, "w") as tex_file:
        write_param_table_header(tex_file, table_type="priors_table")
        # Write one row per parameter
        for prior_dict in all_priors:
            param = prior_dict["param_name"]

            if param not in params_descriptions:
                print(
                    f"Warning: parameter {param} won't be in the priors table as there is no description available"
                )
                continue

            param_info = params_descriptions[param]
            display_value = get_prior_display(prior_dict, table_type="priors_table")
            # write table row
            table_row = f"\\hline {param_info['full_name']} & {display_value} & "
            if "rationale" in param_info:
                table_row += param_info["rationale"]
            tex_file.write("\t " + table_row + " \n")

        # Finish up the table
        tex_file.write("\n \\hline \n \\end{longtable}")


def write_param_table_header(tex_file, table_type="main_table"):
    """
    :param tex_file: an open file
    :param table_type: "main_table" or "priors_table"
    """
    central_column_name = "Value" if table_type == "main_table" else "Distribution"

    # Write the table header
    tex_file.write(
        "\\begin{longtable}[ht]{| >{\\raggedright}p{4cm} | >{\\raggedright}p{3cm} | p{6.8cm} |} \n"
    )
    tex_file.write("\t \hline \n")
    tex_file.write(f"\t Parameter & {central_column_name} & Rationale \\\ \n")
    tex_file.write("\t \endfirsthead \n")
    tex_file.write("\t \\multicolumn{3}{c}{continuation of parameters table} \\\ \n ")
    tex_file.write("\t \endhead \n \n")


def get_fixed_param_display(project, param, param_info):
    # read raw parameter value
    value = read_param_value_from_string(project.param_set.baseline.to_dict(), param)

    # apply potential multiplier
    multiplier = 1 if "multiplier" not in param_info else param_info["multiplier"]
    display_value = multiplier * value

    # round up the value
    if "n_digits" in param_info:
        display_value = round(display_value, param_info["n_digits"])

    # convert to string
    display_value = str(display_value)

    # add potential unit
    if "unit" in param_info:
        display_value += f" {param_info['unit']}"

    return display_value


def get_prior_display(prior_dict, table_type="main_table"):
    """
    Returns text describing a prior distribution
    :param prior_dict: prior dictionary
    :param table_type: "main_table" or "priors_table"
    :return: string
    """
    display = "Calibration parameter \\newline " if table_type == "main_table" else ""

    distri_type = prior_dict["distribution"]
    distri_name = (
        DISTRIBUTION_TRANSLATE[distri_type]
        if distri_type in DISTRIBUTION_TRANSLATE
        else distri_type
    )
    display += f"{distri_name} distribution \\newline "

    distri_params = prior_dict["distri_params"]
    display += f"parameters: [{distri_params[0]}, {distri_params[1]}]"

    if "trunc_range" in prior_dict:
        support = prior_dict["trunc_range"]
        display += f" \\newline support: [{support[0]}, {support[1]}]"

    return display

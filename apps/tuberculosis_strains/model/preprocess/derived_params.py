

def get_derived_params(params):
    # set reinfection contact rate parameters
    for state in ["latent", "recovered"]:
        params["contact_rate_from_" + state] = (
                params["contact_rate"] * params["rr_infection_" + state]
        )

    return params
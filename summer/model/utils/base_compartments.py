def replicate_compartment(
    n_replications,
    current_compartments,
    compartment_stem,
    infectious_compartments,
    initial_populations,
    infectious=False,
    infectious_seed=0.0,
):
    """
    Implements n repeated sequential compartments of a certain type
    Also returns the names of the infectious compartments and the initial population evenly distributed across the
    replicated infectious compartments.

    :params:
        n_replications: int
            number of times compartment is to be repeated
        current_compartments: list
            all the compartments implemented up to this point
        compartment_stem: str
            name of compartment that needs to be replicated
        infectious_compartments: list
            all the infectious compartments implemented up to this point
        initial_population: float

        infectious: bool
            whether the compartment currently being replicated is infectious
        infectious_seed: float
            number of persons to contribute to the infectious seed
    :return:
        updated list of model compartments
        updated list of infectious compartments
        updated dictionary of initial conditions
    """

    # Add the compartment names to the working list of compartments
    compartments_to_add = (
        [compartment_stem]
        if n_replications == 1
        else [compartment_stem + "_" + str(i_comp + 1) for i_comp in range(n_replications)]
    )

    # Add the compartment names to the working list of infectious compartments, if the compartment is infectious
    infectious_compartments_to_add = compartments_to_add if infectious else []

    # Add the infectious population to the initial conditions
    if infectious_seed == 0.0:
        init_pop = {}
    elif n_replications == 1:
        init_pop = {compartment_stem: infectious_seed}
    else:
        init_pop = {
            compartment_stem + "_" + str(i_infectious + 1): infectious_seed / float(n_replications)
            for i_infectious in range(n_replications)
        }
    initial_populations.update(init_pop)

    return (
        current_compartments + compartments_to_add,
        infectious_compartments + infectious_compartments_to_add,
        initial_populations,
    )



def get_cdr_func(detect_prop: float):
    """
    Just need to replace this with the actual function, that will be equivalent to the old Covid model

    Args:
        detect_prop: Currently just a single value representing the case detection rate over time

    Returns:
        The case detection rate function of time

    """

    def cdr_func(time):
        return detect_prop

    return cdr_func

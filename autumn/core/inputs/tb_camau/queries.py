from autumn.core.inputs.database import get_input_db


def get_camau_birth_rate():
    """
    Returns Ca Mau's birth rate
    """

    input_db = get_input_db()
    df = input_db.query(
        "tb_camau",
        columns=["year", "crude_birth_rate"],
    )
    df.dropna(how="any", inplace=True)
    birth_year = df.year.to_numpy()
    birth_rate = df.crude_birth_rate.to_numpy()

    return birth_year, birth_rate


def get_camau_death_rate():
    """
    Returns Ca Mau's death rate
    """

    input_db = get_input_db()
    df = input_db.query(
        "tb_camau",
        columns=["year", "crude_death_rate"],
    )
    df.dropna(how="any", inplace=True)
    death_year = df.year.to_numpy()
    death_rate = df.crude_death_rate.to_numpy()

    return death_year, death_rate

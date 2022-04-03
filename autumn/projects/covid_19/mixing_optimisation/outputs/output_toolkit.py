from autumn import db


def load_derived_output(database_path, output_name):
    df = db.load.load_derived_output_tables(database_path, output_name)[0]
    return df

import os
import pandas as pd
from sqlalchemy import create_engine


class Database:
    """
    Interface to access data stored in a SQLite database.
    """

    def __init__(self, database_name):
        self.database_name = database_name
        self.engine = get_sql_engine(database_name)

    def db_query(self, table_name, column="*", conditions=[]):
        """
        method to query table_name

        :param table_name: str
            name of the database table to query from
        :param conditions: str
            list of SQL query conditions (e.g. ["Scenario='1'", "idx='run_0'"])
        :param value: str
            value of interest with filter column
        :param column:

        :return: pandas dataframe
            output for user
        """
        query = "SELECT %s FROM %s" % (column, table_name)
        if len(conditions) > 0:
            query += " WHERE"
            for condition in conditions:
                query += " " + condition
        query += ";"
        return pd.read_sql_query(query, con=self.engine)


def get_sql_engine(db_path: str):
    """Gets SQL Alchemy databas engine. Mocked out in testing"""
    rel_db_path = os.path.relpath(db_path)
    return create_engine(f"sqlite:///{rel_db_path}", echo=False)

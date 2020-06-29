import os
import pandas as pd
from sqlalchemy import create_engine


class Database:
    """
    Interface to access data stored in a SQLite database.
    """

    def __init__(self, database_path):
        self.database_path = database_path
        self.engine = get_sql_engine(database_path)

    def get_size_mb(self):
        """
        Returns database size in MB.
        """
        query = (
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();"
        )
        size_bytes = self.engine.execute(query).first()[0]
        return size_bytes / 1024 / 1024

    def table_names(self):
        return self.engine.table_names()

    def delete_everything(self):
        """
        Deletes and re-creates the database file.
        """
        try:
            os.remove(self.database_path)
        except FileNotFoundError:
            pass

        self.engine = get_sql_engine(self.database_path)

    def dump_df(self, table_name: str, dataframe: pd.DataFrame):
        dataframe.to_sql(table_name, con=self.engine, if_exists="append", index=False)

    def query(self, table_name, column="*", conditions=[]):
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
        if type(column) is list:
            column_str = ",".join(column)
        else:
            column_str = column

        query = f"SELECT {column_str} FROM {table_name}"
        if len(conditions) > 0:
            condition_chain = " AND ".join(conditions)
            query += f" WHERE {condition_chain}"
        query += ";"
        return pd.read_sql_query(query, con=self.engine)


def get_sql_engine(db_path: str):
    """Gets SQL Alchemy databas engine. Mocked out in testing"""
    rel_db_path = os.path.relpath(db_path)
    return create_engine(f"sqlite:///{rel_db_path}", echo=False)

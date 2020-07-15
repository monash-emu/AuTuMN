import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from pandas.util import hash_pandas_object

logger = logging.getLogger(__name__)


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

    def get_hash(self):
        """
        Returns a hash of the database contents
        """
        table_hashes = [
            hash_pandas_object(self.query(table_name)).mean() for table_name in self.table_names()
        ]
        db_hash = sum(table_hashes)
        return f"{db_hash:0.0f}"

    def table_names(self):
        return self.engine.table_names()

    def column_names(self, table_name):
        return [c[1] for c in self.engine.execute(f"PRAGMA table_info({table_name})")]

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
        df = pd.read_sql_query(query, con=self.engine)

        # Backwards compatibility fix for old column names with square brackets
        column_names = self.column_names(table_name)
        renames = {}
        for column_name in column_names:
            # Assume column is named something like foo.bar[-1]
            # and we will rename to foo.bar(-1)
            if "[" in column_name:
                logger.info("Cleaning square brackets from column %s", column_name)
                df_name = column_name.split("[")[0]
                new_name = column_name.replace("[", "(").replace("]", ")")
                renames[df_name] = new_name

        if renames:
            df.rename(columns=renames, inplace=True)

        return df


def get_sql_engine(db_path: str):
    """Gets SQL Alchemy databas engine. Mocked out in testing"""
    rel_db_path = os.path.relpath(db_path)
    return create_engine(f"sqlite:///{rel_db_path}", echo=False)

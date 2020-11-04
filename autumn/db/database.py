import os
import logging
from abc import abstractmethod, ABC
from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import create_engine
from pandas.util import hash_pandas_object

logger = logging.getLogger(__name__)


class BaseDatabase(ABC):
    """
    Interface to access data stored somewhere.
    """

    @abstractmethod
    def __init__(self, database_path: str):
        """Sets up the database"""

    @staticmethod
    @abstractmethod
    def is_compatible(database_path: str) -> bool:
        """Returns True if the database is compatible with the given path"""

    @abstractmethod
    def table_names(self):
        """Returns a list of table names"""

    @abstractmethod
    def delete_everything(self):
        """Deletes and recreates the db."""

    @abstractmethod
    def dump_df(self, table_name: str, dataframe: pd.DataFrame):
        """Writes a dataframe to a table. Appends if the table already exists."""

    @abstractmethod
    def query(
        self, table_name: str, columns: List[str] = [], conditions: Dict[str, Any] = {}
    ) -> pd.DataFrame:
        """Returns a dataframe"""


class FeatherDatabase(BaseDatabase):
    """
    Interface to access data stored in a Feather "database".
    https://arrow.apache.org/docs/python/feather.html
    """

    def __init__(self, database_path: str):
        """Sets up the database"""
        self.database_path = database_path
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        elif not os.path.isdir(database_path):
            raise ValueError(f"FeatherDatabase requires a folder as a target: {database_path}")

        feather_files = os.listdir(database_path)
        if not all(f.endswith(".feather") for f in feather_files):
            raise ValueError(
                f"FeatherDatabase target must contain only .feather files, got: {feather_files}"
            )

    @staticmethod
    def is_compatible(database_path: str) -> bool:
        """Returns True if the database is compatible with the given path"""
        if database_path.endswith(".db"):
            return False
        elif not os.path.exists(database_path):
            return True
        elif not os.path.isdir(database_path):
            return False
        elif not all(f.endswith(".feather") for f in os.listdir(database_path)):
            return False
        else:
            return True

    def table_names(self):
        """Returns a list of table names"""
        return [f.replace(".feather", "") for f in os.listdir(self.database_path)]

    def delete_everything(self):
        """Deletes and recreates the db."""
        for fname in os.listdir(self.database_path):
            fpath = os.path.join(self.database_path, fname)
            os.remove(fpath)

    def dump_df(self, table_name: str, df: pd.DataFrame):
        """Writes a dataframe to a table. Appends if the table already exists."""
        fpath = os.path.join(self.database_path, f"{table_name}.feather")
        write_df = df
        if os.path.exists(fpath):
            # Read in existing dataframe and then append to the end of it.
            # This could be slow so ideally don't do this.
            orig_df = pd.read_feather(fpath)
            write_df = orig_df.append(df)

        write_df.columns = write_df.columns.astype(str)
        write_df.to_feather(fpath)

    def query(
        self, table_name: str, columns: List[str] = [], conditions: Dict[str, Any] = {}
    ) -> pd.DataFrame:
        """Returns a dataframe"""
        fpath = os.path.join(self.database_path, f"{table_name}.feather")
        df = pd.read_feather(fpath)
        if columns:
            df = df[columns]

        for k, v in conditions.items():
            df = df[df[k] == v]

        if columns or conditions:
            df = df.copy()

        return df


class Database(BaseDatabase):
    """
    Interface to access data stored in a SQLite database.
    """

    def __init__(self, database_path):
        self.database_path = database_path
        self.engine = get_sql_engine(database_path)

    @staticmethod
    def is_compatible(database_path: str) -> bool:
        """Returns True if the database is compatible with the given path"""
        return database_path.endswith(".db")

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

    def dump_df(self, table_name: str, df: pd.DataFrame):
        df.to_sql(table_name, con=self.engine, if_exists="append", index=False)

    def query(
        self, table_name: str, columns: List[str] = [], conditions: Dict[str, Any] = {}
    ) -> pd.DataFrame:
        """
        method to query table_name
        """
        column_str = ",".join(columns) if columns else "*"
        query = f"SELECT {column_str} FROM {table_name}"
        if len(conditions) > 0:
            c_exps = []
            for k, v in conditions.items():
                if v is None:
                    c_exp = f"{k} IS NULL"
                elif type(v) is str:
                    c_exp = f"{k}='{v}'"
                else:
                    c_exp = f"{k}={v}"

                c_exps.append(c_exp)

            condition_chain = " AND ".join(c_exps)
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


DATABASE_TYPES = [Database, FeatherDatabase]


def get_database(database_path: str) -> BaseDatabase:
    """Returns the correct kind of BaseDatabase for a given database path"""
    for db_type in DATABASE_TYPES:
        if db_type.is_compatible(database_path):
            return db_type(database_path)

    raise ValueError(f"Could not find a database that works with path: {database_path}")


def convert_database(orig_db: BaseDatabase, new_db_cls, new_db_path: str) -> BaseDatabase:
    """Writes all data as a new_db_cls file, returning the new new_db_cls database"""
    new_db = new_db_cls(new_db_path)
    for table_name in orig_db.table_names():
        new_db.dump_df(table_name, orig_db.query(table_name))


def get_sql_engine(db_path: str):
    """Gets SQL Alchemy databas engine. Mocked out in testing"""
    rel_db_path = os.path.relpath(db_path)
    return create_engine(f"sqlite:///{rel_db_path}", echo=False)

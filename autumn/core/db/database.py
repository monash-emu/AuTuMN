import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
import pyarrow
import pyarrow.parquet as parquet
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class BaseDatabase(ABC):
    """
    Interface to access data stored somewhere.
    """

    @abstractmethod
    def __init__(self, database_path: str):
        """Sets up the database"""

    @classmethod
    @abstractmethod
    def is_compatible(cls, database_path: str) -> bool:
        """Returns True if the database is compatible with the given path"""

    @abstractmethod
    def table_names(self):
        """Returns a list of table names"""

    @abstractmethod
    def delete_everything(self):
        """Deletes and recreates the db."""

    @abstractmethod
    def dump_df(self, table_name: str, dataframe: pd.DataFrame, append=True):
        """Writes a dataframe to a table. Appends if the table already exists."""

    @abstractmethod
    def query(
        self, table_name: str, columns: List[str] = [], conditions: Dict[str, Any] = {}
    ) -> pd.DataFrame:
        """Returns a dataframe"""


class FileDatabase(BaseDatabase, ABC):
    """
    A database that uses one file per dataframe
    """

    extension: str = None  # Eg. .feather

    def __init__(self, database_path: str):
        """Sets up the database"""
        self.database_path = database_path
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        elif not os.path.isdir(database_path):
            raise ValueError(f"FileDatabase requires a folder as a target: {database_path}")

        files = os.listdir(database_path)
        if not all(f.endswith(self.extension) for f in files):
            raise ValueError(
                f"FileDatabase target must contain only {self.extension} files, got: {files}"
            )

    @classmethod
    def is_compatible(cls, database_path: str) -> bool:
        """Returns True if the database is compatible with the given path"""
        path_exists = os.path.exists(database_path)
        if not path_exists:
            # Non existent database compatible with all types.
            return True

        is_path_dir = os.path.isdir(database_path)
        if not is_path_dir:
            # Only works with directorys with files inside.
            return False

        is_correct_extension = all(f.endswith(cls.extension) for f in os.listdir(database_path))
        return is_correct_extension

    def table_names(self):
        """Returns a list of table names"""
        return [f.replace(self.extension, "") for f in os.listdir(self.database_path)]

    def delete_everything(self):
        """Deletes and recreates the db."""
        for fname in os.listdir(self.database_path):
            fpath = os.path.join(self.database_path, fname)
            os.remove(fpath)

    def dump_df(self, table_name: str, df: pd.DataFrame, append=True):
        """
        Writes a dataframe to a table. Appends if the table already exists.
        It is much more memory efficient to use append_df if possible (eg. ParquetDatabase).
        """
        fpath = os.path.join(self.database_path, f"{table_name}{self.extension}")
        if os.path.exists(fpath) and append:
            # Read in existing dataframe and then append to the end of it.
            # This could be slow so ideally don't do this.
            orig_df = self.read_file(fpath)
            write_df = orig_df.append(df)
        else:
            write_df = df.copy()
        try:
            write_df.reset_index(drop=True, inplace=True)
        except ValueError:
            pass

        write_df.columns = write_df.columns.astype(str)
        self.write_file(fpath, write_df)

    def query(
        self, table_name: str, columns: List[str] = [], conditions: Dict[str, Any] = {}
    ) -> pd.DataFrame:
        """Returns a dataframe"""
        fpath = os.path.join(self.database_path, f"{table_name}{self.extension}")
        
        if conditions:
            if columns:
                tmp_columns = columns + list(conditions)
                df = self.read_file(fpath, tmp_columns)
            else:
                df = self.read_file(fpath)
            for k, v in conditions.items():
                if v is None:
                    df = df[df[k].isnull()]
                else:
                    df = df[df[k] == v]
    
            if columns:
                df = df[columns]
            
            df = df.copy()
        else:
            df = self.read_file(fpath, columns)

        return df

    @abstractmethod
    def read_file(self, path: str, columns: List[str]) -> pd.DataFrame:
        """How to read a file from disk"""

    @abstractmethod
    def write_file(self, path: str, df: pd.DataFrame):
        """How to write a file to disk"""


class FeatherDatabase(FileDatabase):
    """
    Interface to access data stored in a Feather "database".
    https://arrow.apache.org/docs/python/feather.html
    """

    extension = ".feather"

    def read_file(self, path: str, columns: List[str] = []) -> pd.DataFrame:
        """How to read a file from disk"""
        return pd.read_feather(path, columns=columns or None)

    def write_file(self, path: str, df: pd.DataFrame):
        """How to write a file to disk"""
        df.to_feather(path)


class ParquetDatabase(FileDatabase):
    """
    Interface to access data stored in a Parquet "database".
    https://arrow.apache.org/docs/python/feather.html
    """

    extension = ".parquet"

    def __init__(self, database_path: str):
        super().__init__(database_path)
        self.writers = {}

    def read_file(self, path: str, columns: List[str] = []) -> pd.DataFrame:
        """How to read a file from disk"""
        return pd.read_parquet(path, columns=columns or None)

    def write_file(self, path: str, df: pd.DataFrame):
        """How to write a file to disk"""
        df.to_parquet(path)

    def append_df(self, table_name: str, df: pd.DataFrame):
        """
        Writes a dataframe to a table. Appends if the table already exists.
        This is more memory efficient than using FileDatabase.dump_df, but you must call
        close() when you're done (probably).
        """
        table = pyarrow.Table.from_pandas(df)
        fpath = os.path.join(self.database_path, f"{table_name}{self.extension}")
        if not self.writers.get(table_name):
            self.writers[table_name] = parquet.ParquetWriter(fpath, table.schema)

        self.writers[table_name].write_table(table)

    def close(self):
        """
        Close writers after appending.
        """
        for writer in self.writers.values():
            writer.close()

        self.writers = {}


class Database(BaseDatabase):
    """
    Interface to access data stored in a SQLite database.
    """

    def __init__(self, database_path):
        self.database_path = database_path
        self.engine = get_sql_engine(database_path)
        self._cache = {}

    @classmethod
    def is_compatible(cls, database_path: str) -> bool:
        """Returns True if the database is compatible with the given path"""
        return database_path.endswith(".db")

    def table_names(self) -> List[str]:
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

    def dump_df(self, table_name: str, df: pd.DataFrame, append=True):
        exists_mode = 'append' if append else 'replace'
        df.to_sql(table_name, con=self.engine, if_exists=exists_mode, index=False)

    def query(
        self, table_name: str, columns: List[str] = [], conditions: Dict[str, Any] = {},
        as_copy=True
    ) -> pd.DataFrame:
        """
        method to query table_name

        as_copy can be False if needed for performance, but only if you promise not to modify the returned data...
        """
        def sanitize(name):
            if " " in name:
                return f"`{name}`"
            else:
                return name

        columns = [sanitize(c) for c in columns]

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

        if query not in self._cache:
            self._cache[query] = df = pd.read_sql_query(query, con=self.engine)
        
        df = self._cache[query]

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

        if as_copy:
            df = df.copy()

        return df


DATABASE_TYPES = [Database, FeatherDatabase, ParquetDatabase]


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

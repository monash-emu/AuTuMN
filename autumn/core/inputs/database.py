import logging
import os

from autumn.core.db import Database
from autumn.core.db.database import ParquetDatabase, get_database
from autumn.core.utils.timer import Timer
from autumn.settings import INPUT_DATA_PATH

from .covid_au.preprocess import preprocess_covid_au
from .covid_bgd.preprocess import preprocess_covid_bgd
from .covid_btn.preprocess import preprocess_covid_btn
from .covid_lka.preprocess import preprocess_covid_lka
from .covid_mmr.preprocess import preprocess_covid_mmr
from .covid_mys.preprocess import preprocess_covid_mys
from .covid_phl.preprocess import preprocess_covid_phl
from .covid_survey.preprocess import preprocess_covid_survey
from .covid_vnm.preprocess import preprocess_covid_vnm
from .demography.preprocess import preprocess_demography
from .gisaid_voc.preprocess import preprocess_covid_gisaid
from .mobility.preprocess import preprocess_mobility
from .owid.preprocess import preprocess_our_world_in_data
from .school_closure.preprocess import preprocess_school_closure
from .social_mixing.preprocess import preprocess_social_mixing
from .tb_kir.preprocess import preprocess_tb_kir

logger = logging.getLogger(__name__)

_input_db = None

INPUT_DB_PATH = os.path.join(INPUT_DATA_PATH, "db")


def get_input_db():
    global _input_db
    if _input_db:
        return _input_db
    else:
        _input_db = build_input_database()
        return _input_db


def build_input_database(rebuild: bool = False):
    """
    Builds the input database from scratch.
    If force is True, build the database from scratch and ignore any previous hashes.
    If force is False, do not build if it already exists,
    and crash if the built database hash does not match.

    If rebuild is True, then we force rebuild the database, but we don't write a new hash.

    Returns a Database, representing the input database.
    """
    if os.path.exists(INPUT_DB_PATH) and not rebuild:
        input_db = get_database(INPUT_DB_PATH)
    else:
        logger.info("Building a new database.")
        input_db = ParquetDatabase(INPUT_DB_PATH)

        with Timer("Deleting all existing data."):
            input_db.delete_everything()

        with Timer("Ingesting COVID NT data."):
            preprocess_covid_au(input_db)

        with Timer("Ingesting COVID PHL data."):
            preprocess_covid_phl(input_db)

        with Timer("Ingesting COVID MYS data."):
            preprocess_covid_mys(input_db)

        with Timer("Ingesting COVID LKA data."):
            preprocess_covid_lka(input_db)

        with Timer("Ingesting COVID MMR data."):
            preprocess_covid_mmr(input_db)

        with Timer("Ingesting COVID BGD data."):
            preprocess_covid_bgd(input_db)

        with Timer("Ingesting COVID BTN data."):
            preprocess_covid_btn(input_db)

        with Timer("Ingesting TB KIR data."):
            preprocess_tb_kir(input_db)

        with Timer("Ingesting COVID survey data"):
            preprocess_covid_survey(input_db)

        with Timer("Ingesting Our World in Data data."):
            preprocess_our_world_in_data(input_db)

        with Timer("Ingesting school closure data."):
            preprocess_school_closure(input_db)

        with Timer("Ingesting demography data."):
            country_df = preprocess_demography(input_db)

        with Timer("Ingesting gisaid data."):
            preprocess_covid_gisaid(input_db)

        with Timer("Ingesting social mixing data."):
            preprocess_social_mixing(input_db, country_df)

        with Timer("Ingesting gisaid data."):
            preprocess_covid_gisaid(input_db)

    return input_db

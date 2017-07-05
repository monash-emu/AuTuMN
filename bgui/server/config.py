SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://optima:optima@localhost:5432/baseoptima'
SECRET_KEY = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'
SAVE_FOLDER = '/tmp/baseoptima'
CELERY_BROKER_URL = 'redis://localhost:6379/1'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/1'
REDIS_URL = CELERY_BROKER_URL

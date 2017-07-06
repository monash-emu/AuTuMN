# Introduction to the BASE webserver

- the main webserver is `api.py`, a `python` application, written in the `flask` framework.
 `api.py` defines the way URL requests are converted into HTTP responses to the
  webclient
- `flask` is a micro-framework to provide libraries to build `api.py`
- `api.py` stores projects in a `postgres` database, a robust high-performant database. 
  It uses the `sqlalchemy` python library to interface with the database, 
  the database schema, tables and entries. You can directly interrogate the `postgres` 
  database using the standard `postgres` command-line tools, such as `psql`. The
  database schema is stored in `server/webapp/dbmodel.py`.
- `bin/run_server.py` uses `twisted` to bridge `api.py` to the outgoing port of your computer using the 
  `wsgi` specification. `run_server.py` also serves the client js files from the `client` folder 
- to carry out parallel simulations, `server/webapp/tasks.py` is a  `celery` daemon that listens for jobs
  from `api.py`, which is communicated through an in-memory/intermittent-disk-based `redis`
  database.

## Installing the server

_On Mac_

1. install `brew` as it is the best package-manager for Mac OSX:
    - `ruby -e “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)”`
2. install `python` and all the required third-party `python` modules:
    - `brew install python`
    - to make sure `python` can find the Optima `<root>` directory, in the top `<root>` folder, run: 
        `python setup.py develop` 
    - to load all the `python` modules needed (such as `celery`, `flask`, `twisted` etc), in the `<root>/server` folder, run:
        `pip install -r localrequirements`
3. install the ruby `lunchy` controller for MacOSX daemons
    - `brew install ruby`
    - `sudo gem install lunchy`
4. install the `redis` and `postgres` database daemons:
    - `brew install postgres redis`
    - launch them: `lunchy start redis postgres`

_On Ubuntu:_

1. `sudo apt-get install redis-server`

## Configuring the webserver

Next, we set up the databases, brokers, url's that `api.py` will we use. We do this through the `config.py` file in the `<root>/server` folder. Here's an example of a `config.py` file:

```
SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://optima:optima@localhost:5432/optima'
SECRET_KEY = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'
UPLOAD_FOLDER = '/tmp/uploads'
CELERY_BROKER_URL = 'redis://localhost:6379'
CELERY_RESULT_BACKEND = 'redis://localhost:6379'
REDIS_URL = CELERY_BROKER_URL
MATPLOTLIB_BACKEND = "agg"
```

You can choose which ports to use, the name of the databases. Matplotlib is the key python library 
that generates the graphs, it must be set to the "agg" backend, otherwise it can cause GUI crashes with
the windowing system of your computer.

The port that `api.py` is run on, is set in `<root>/server/_twisted_wsgi.py`, which by default is 8080.


## Running the webserver

If you haven't already launched `redis` and `postgres`, launch them:  

`lunchy start redis postgres`

Then, from the `<root>/bin` directory:

- launch the webserver `./start_server.sh` 
- launch the parallel-processing daemon `./start_celery.sh`

__!__ Note: whenever you make a change that could affect a celery task, you need to restart it manually.


## Files

- `api.py` - the webserver
- `config.py` - the configuration of the postgres/redis databases and ports for the webserver
- `requirements.txt` - python modules to be installed

In `server/webapp`:

- `handlers.py` - the URL handlers are defined here. The handlers only deal with JSON-data structures,
  UIDs, and upload/download files
- `dbconn.py` - this is a central place to store references to the postgres and redis database
- `dbmodel.py` - the adaptor to the Postgres database with objects that map onto the database tables
- `tasks.py` - this is a conceptually difficult file - it defines both the daemons
   and the tasks run by the daemon. This file is called by `api.py` as entry point to talk to `celery`,
   and is run separately as the `celery` task-manager daemon.


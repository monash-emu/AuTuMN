# 1. Overview

This README describes the steps involved in installing and running Optima. **Follow the instructions in the "Quick start guide" immediately below.** Unless you're a developer, you won't need to follow the rest of the instructions.

## 1.1 Quick start guide

0. Download and install Anaconda, using default options (https://store.continuum.io/cshop/anaconda/). **Make sure you download Python 2.7, not 3.5.**

0. Sign up for an account on GitHub (http://github.com) with a free plan.

0. Download and install the GitHub app (http://desktop.github.com) on Windows or Mac, or `sudo apt-get install git` on Linux.

0. Go to the Optima GitHub page and click on the small button next to "Download ZIP":
![](http://optimamodel.com/figs/optima-github-button.png)
(or use `git clone` if on Linux)

0. Finally, set up the Python path:
    0. Run Spyder (part of Anaconda)
    0. Under the “Tools” (Linux and Windows) or “python” (under Mac) menu, go to “PYTHONPATH Manager”
    0. Select the Optima folder (e.g. `C:\Users\Alice\GitHub\Optima` on Windows) and click OK.

0. To check that everything works:
  0. Run Spyder (e.g. Anaconda -> Spyder from the Start Menu in Windows; Anaconda Launcher or `spyder` from the Terminal on Mac or Linux)
  0. Open a new Python console (Console -> Python console)
  0. In the Spyder editor (File -> Open), go to the `Optima/tests` folder and open `simple.py`
  0. Run (F5, or select "Run" from the "Run" menu)
  0. You should see a figure appear -- note that it might appear in the console (if you're using IPython) or in a separate window but minimized.



# 2. Optima model setup

## 2.1 Quick start installation

To install, run `python setup.py develop` in the root repository directory. This will add Optima to the system path. Optima can then be used via Python.

To uninstall, run `python setup.py develop --uninstall`.

Note: do **not** use `python setup.py install`, as this will copy the source code into your system Python directory, and you won't be able to modify or update it easily.


## 2.2 Detailed instructions

### 2.2.1 Preliminaries

0. Make sure you have a GitHub (http://github.com) account, plus either git or the GitHub app (http://desktop.github.com) -- which it looks like you do, if you're reading this :)
0. Clone the Optima repository: https://github.com/optimamodel/Optima.git
0. Make sure you have a version of scientific Python. Easiest to set up is probably Anaconda (https://store.continuum.io/cshop/anaconda/).

### 2.2.2 Dependencies

If you use Anaconda, everything should be taken care of, except possibly `pyqt4`, which is needed for the Python GUI.

If you don't want to use Anaconda, you'll need to install the dependencies yourself (via e.g. `pip install`). If you install the latest versions of `numpy`, `matplotlib`, `xlrd`, `xlsxwriter`, and `pyqt4`, and `mpld3`, all of the backend should work.

The full list of requirements (including for the frontend) is given in `server/requirements.txt`. However, note that `run.sh` will create a virtual environment with these packages even if you don't have them available on your system.

### 2.2.3 Python path

The last step is to make sure that Optima is available on the Python path. There are several ways of doing this:

 0. **Option 1: Spyder path**
    0. Run Spyder (part of Anaconda)
    0. Under the “Tools” (Linux and Windows) or “python” (under Mac) menu, go to “PYTHONPATH Manager”
    0. Select the Optima folder (e.g. `C:\Users\Alice\GitHub\Optima` on Windows) and click OK.
 0. **Option 2: modify system path**
    0. **Option 2A** (all operating systems): Go to the Optima root folder (in a terminal on Mac or Linux; in a command prompt [cmd.exe] in Windows) and run
    `python setup.py develop`
    Note: if Spyder does not use the system Python (which can happen in some cases), this will not work. In this case:
       0. Inside a Spyder console, type
          `import sys; sys.executable`
       0. Replace the above command with the location of this executable, e.g.
          `/software/anaconda/bin/python setup.py develop`
    0. **Option 2B** (Linux, Mac only): Add the Optima folder to `~/.bashrc` or `~/.bash_profile`, e.g.
    `export PYTHONPATH=$PYTHONPATH:/users/alice/github/optima`
    [NB: if you don't use `bash`, you are probably a hacker and don't need these instructions.]
    0. **Option 2C** (Windows only): search for “variables” from the Start Menu; the option should be called something like “Edit environment variables for your account”. Under “user variables”, you should see “PYTHONPATH” listed. Add the folder for the Optima repository, e.g.
    `C:\Users\Alice\GitHub\Optima`
    If there are already things on the Python path, add this to the end separated by a semicolon and no space, e.g.
    `C:\Anaconda2\Library\bin;C:\Users\Alice\GitHub\Optima`

### 2.3 Verification/usage

If you followed the steps correctly, you should be able to run
`import optima`
from a Python console (either the system console or the Spyder console)

For usage examples, see the scripts in the `tests` folder. In particular, `testworkflow.py` shows a typical usage example.





# 3. Database setup

*For further details, see server/db/README.md*

## 3.1 Installing PostgreSQL

On mac, install the `postgres` software with:

    brew install postgres

On Linux, use

    sudo apt-get install install postgres

Then you create the default database store:

    initdb /usr/local/var/postgres -E utf8

To run the `postgres` daemon in a terminal:

```bash
postgres -D /usr/local/var/postgresbrew
```

If you want to, you can run the `postgres` daemon with the Mac system daemon manager `launchctl`, or via the ruby wrapper for `lunchy`.


## 3.2 Setting up the optima database

For the development environment setup Optima needs to use a Postgres database created using:

- name: `optima`
- host: `localhost`
- port: `5432`
- username: `optima`
- password: `optima`

Warning: the migrate scripts requires a user called `postgres`. This may not have been installed for you. One way to do this is to switch the user on your system `postgres` before building the database:

    sudo su postgres

Alternatively, you can create the `postgres` user directly:

    createuser postgres -s

You will first need to install the python database migration tools:

```bash
pip install sqlalchemy-migrate psycopg2
```

Then to create the optima database, use these commands *from the root Optima directory* as `migrate` needs to find the migration scripts:

```bash
createdb optima # Create Optima database -- for run.sh
createdb test # Create test database -- for test.sh
createuser optima -P -s # with password optima
createuser test -P -s # with password test
migrate version_control postgresql://optima:optima@localhost:5432/optima server/db/ # Allow version control
migrate upgrade postgresql://optima:optima@localhost:5432/optima server/db/ # Run the migrations to be safe
```

The scripts require that the `optima` user is a superuser. To check this:

```bash
psql -d optima -c "\du"
```

You should be able to see the users `optima` and `postgres`, and they are set to superusers. If not, to set `optima` to superuser:

```bash
psql -d optima -c "ALTER USER optima with SUPERUSER;"
```



# 4. Client setup

*For further details, see client/README.md*

This has been made using seed project [ng-seed](https://github.com/StarterSquad/ngseed/wiki)

## 4.1 Installing the client

Run script:

    client/clean_build.sh

In case you face issue in executing ./clean_build.sh you can alternatively execute commands:

1. `npm install`
2. `npm -g install bower (if you do not have bower already globally installed)`
3. `npm -g install gulp (if you do not have gulp already globally installed)`
4. Create file `client/source/js/version.js` and add this content to it:

        define([], function () { return 'last_commit_short_hash'; });

    (Where last_commit_short_hash is short hash for the last commit).



# 5. Server setup

*For further details, see server/README.md*


## 5.1 Installation

This component requires:

- [pip](http://pip.readthedocs.org/en/latest/installing.html) - python packaging manager
- [VirtualEnv](http://virtualenv.readthedocs.org/en/latest/) - python environment manager
- [tox](http://http://tox.readthedocs.org/) - virtualenv manager
- [PostgreSQL](http://www.postgresql.org/download/)  - relational database
- [Redis](http://redis.io/) - memory caching
- [Celery](http://redis.io/) - distributed task queue

To install the Redis server:

_On Linux_:

    sudo apt-get install redis-server

_On Mac_:

    brew install redis
    gem install lunchy # a convenient daemon utility script
    ln -sfv /usr/local/opt/redis/*.plist ~/Library/LaunchAgents
    lunchy start redis

Copy over the setup:

    cp server/config.py.example server/config.py

NOTE: config.example.py (the reference config) can be changed (e.g. new settings added or old settings removed). If you have problems with running Optima locally, look at the reference config file and compare it with your version.

Then to run the server, there are two options -- directly (self-managed environment like Anaconda) or through a virtualenv (if you are a developer).

_Using the scripts directly (e.g. prod/Anaconda)_:

Make sure you have the requirements:

    pip install -r server/requirements.txt

Then run the server in one terminal:

    python bin/run_server.py 8080 # to start on port 8080

...and celery in the other:

    celery -A server.webapp.tasks.celery_instance worker -l info


_Using Virtualenvs (e.g. for development)_:

Install ``virtualenv`` and ``tox``:

    pip install virtualenv tox

Run the server in two separate terminals. These scripts will start Python in a `virtualenv` isolated Python environments.
If you wish to use system installed packages, append `--sitepackages` and it will not reinstall things that are already installed in the Python site packages.
First in one terminal:

    tox -e celery

Then in the other terminal:

    tox -e runserver



## 5.2 Tests

Run the test suite from your server directory:

    ./test.sh

In order to run a single test file and activate logging you can use:

    test.sh /src/tests/project_test.py

You can use `--system` as first argument to `test.sh` in order to use pre-installed system-wide python libraries.

Make sure you have user "test" with the password "test" and database "optima_test" in order to run the tests using database.



# 6. Usage

If all steps have been completed, run ``tox -e runserver`` in the server directory, and then go to `http://optima.dev:8080` in your browser (preferably Chrome). You should see the Optima login screen.

In order to use the application you need to login a registered user. In order to register a new user, visit <http://optima.dev:8080/#/register>, and register using any details.

Happy Optimaing!



# 7. Wisdom

This section contains random pieces of wisdom we have encountered along the way.

## 7.1 Workflows

- Make sure you pull and push from the repository regularly so you know what everyone else is doing, and everyone else knows what you're doing. If your branch is 378 commits behind develop, you're the sucker who's going to have to merge it.
- There is very unclear advice about how to debug Python. It's actually fairly simple: if you run Python in interactive mode (e.g. via Spyder or via `python -i`), then if a script raises an exception, enter this in the console just after the crash:
`import pdb; pdb.pm()`
You will then be in a debugger right where the program crashed. Type `?` for available commands, but it works like how you would expect. Alternatively, if you want to effectively insert a breakpoint into your program, you can do this with
`import pdb; pdb.set_trace()`
No one knows what these mysterious commands do. Just use them.
- For benchmarking/profiling, you can use `tests/benchmarkmodel.py`. It's a good idea to run this and see if your changes have slowed things down considerably. It shows how to use the line profiler; Spyder also comes with a good function-level (but not line) profiler.


## 7.2 Python gotchas

- Do not declare a mutable object in a function definition, e.g. this is bad:
```
def myfunc(args=[]):
  print(args)
```
The arguments only get initialized when the function is declared, so every time this function is used, there will be a single `args` object shared between all of them! Instead, do this:
```
def myfunc(args=None):
  if args is None: args = []
  print(args)
```
- It's dangerous to use `type()`; safer to use `isinstance()` (unless you _really_ mean `type()`). For example,
`type(rand(1)[0])==float`
is `False` because its type is `<type 'numpy.float64'>`; use `isinstance()` instead, e.g.   `isinstance(rand(1)[0], (int, float))`
 will catch anything that looks like a number, which is usually what you _really_ want.

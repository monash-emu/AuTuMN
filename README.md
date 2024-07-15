# AuTuMN

## Note - this project is deprecated; new users should consult the EMU homepage https://monash-emu.github.io/

![](https://github.com/monash-emu/AuTuMN/workflows/Continuous%20Integration/badge.svg)

This is a disease modelling project, written in Python.
It is currently being applied to COVID-19 and tuberculosis. This project is used and maintained by the Monash Univeristy Epidemiological Modelling Unit.

AuTuMN uses the "summer" disease modelling framework, with code [here](https://github.com/monash-emu/summer) which is [documented here](http://summerepi.com/) with examples and an API reference.

See [this guide](./docs/setup.md) for information on how to set up this project.

## Using the command line

All of Autumn's features can be accessed from the command line. You can run commands as follows:

```bash
python -m autumn <YOUR COMMANDS HERE>
```

To see a list of options, try the help prompt:

```bash
python -m autumn --help
```

## Project structure

```
├── .github                 GitHub Actions config
|
├── autumn                  Main project Python codebase
|   ├─ command_line             Command line interface
|   ├─ models                   Generic disease models
|   ├─ notebooks                Example and user specific Jupyter notebooks
|   ├─ projects                 Region specifc projects that use disease models
|   ├─ remote                   Remote server orchestration tasks
|   ├─ settings                 Globally shared constants
|   ├─ tasks                    Remote server pipeline tasks
|   └─ tools                    Globally shared utilities
|
├── data                    Data to be used by the models
|   ├─ inputs                   Input data for the models
|   └─ outputs                  Module run outputs (not in source control)
|
├── docs                    Documentation
|   ├─ schemas                  SQL schemas                   
|   └─ tex_description          Descriptions of the modelling-related files, in TeX
|       
├── scripts                 Utility scripts
├── tests                   Automated tests
├── .gitignore              Files for Git to ignore
├── pyproject.toml          Configuration for tools (eg. Black, pytest)
└── requirements.txt        Python library dependencies
```

## Running an application

You can run all the scenarios for specific application using the `run` command. For example, to run the "malaysia" region of the "covid" model, you can run:

```bash
python -m autumn project run covid_19 malaysia
```

Model run outputs are written to `data/outputs/run`.

Run time depends on the complexity of the model, the simulated time window and the number of scenarios simulated.
For example, the run time of a COVID-19 model with simulation of a single scenario over a period of 12 months is around 4 seconds on a normal desktop computer.

## Running a calibration

You can run a model MCMC calibration as follows

```bash
python -m autumn project calibrate covid_19 malaysia 30 1
python -m autumn project calibrate calibrate MODEL_NAME REGION_NAME MAX_SECONDS RUN_ID
```

For example, to calibrate the malaysia COVID model for 30 seconds you can run:

```bash
python -m apps calibrate covid malaysia 30 0
```

The RUN_ID argument can always be "0" for local use, it doesn't really matter.

Model calibration outputs are written to `data/outputs/calibrate`.

## Running the automated tests

We have a suite of automated tests that verify that the code works. Some of these are rigorous "unit" tests which validate functionality, while others are only "smoke" tests, which verify that the code runs without crashing. These tests are written with [pytest](https://docs.pytest.org/en/stable/).

You are encouraged to run the tests locally before pushing your code to GitHub. Automated tests may be run via [PyCharm](https://www.jetbrains.com/help/pycharm/pytest.html) or via the command line using pytest:

```
pytest -v
```

Tests are also run automatically via [GitHub Actions](https://github.com/monash-emu/AuTuMN/actions) on any pull request or commit to the `master` branch.

## Reading and writing secrets

There are some files that we do not want to share publicly. These are encrypted and use the filename `*.encrypted.*`.

### Reading all encrypted files

To read the encrypted files you will need to know the project password.
Once you have the password, then you can read the files:

```bash
python -m autumn secrets read
```

The decrypted files will have the filename `*.secret.*`. **Do not remove this or change the filename**.

This will throw an error if you enter the incorrect password, or if the read file does not match the hashes stored in `data/secret-hashes.json`.

### Writing all encrypted files

To write a new encrypted file, first name your file `*.secret.*` and then use the CLI to encrypt it:

```bash
python -m autumn secrets write apps/foo/bar.secret.json
```

A new file called `apps/foo/bar.encrypted.json` will be created. You can commit this to GitHub.

## Benchmarking

Automated benchmarks are run on every commit to master using the [github-action-benchmark](https://github.com/rhysd/github-action-benchmark) GitHub Action. These benchmarks are run inside of pytest via `pytest-benchmark` and can be found in `tests/test_benchmarks.py`. You can view the results at [monash-emu.github.io/AuTuMN](http://monash-emu.github.io/AuTuMN/)

## Formatting

The codebase can be auto-formatted using [Black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/).
You can auto-format the code as follows, this will (probably) never break anything:

```
black .
isort . --profile black
```

## Input data

Input data is stored in text format in the `data/inputs/` folder. All input data required to run the app should be stored in this folder, along with a README explaining its meaning and provenance. Input data is preprocessed into an SQLite database, using the `autumn.core.inputs` module. The resulting database is stored at `data/inputs/inputs.db` and is tracked using Git LFS. If you want to add new input data or modify existing data, then:

- add or update the source CSV/XLS files
- adjust the preprocess functions in `autumn.core.inputs` as required
- rebuild the database, commit the new file to source control

To fetch the latest data, run:

```bash
python -m autumn db fetch
```

You will need to ensure that the latest date in all user-specified mixing data params is greater than or equal to the most recent Google Mobility date.

To rebuild the database with new data, run:

```bash
python -m autumn db build
```

Once you are satisfied that all your models work again (run the tests), commit your changes and push up:

- The updated CSV files
- The updated `data/inputs/inputs.encrypted.db` file
- Any required changes to model parameters (eg. dynamic mixing dates)

Note that all of our CSV and XLSX files are stored using [Git Large File Storage](https://docs.github.com/en/github/managing-large-files/versioning-large-files) in GitHub. You can install Git LFS [here](https://git-lfs.github.com/).

## AWS Calibration

We often need to run long, computationally expensive jobs. We are currently using Amazon Web Services (AWS) to do this. The scripts and documentation that allow you to do this can be found in the `remote/aws/` folder. The following jobs are run in AWS:

- Calibration: Using an adaptive Metropolis algorithm to fit the model to observations
- Full model runs: Running all scenarios for all accepted Metropolis parameter sets
- PowerBI processing: Post-processing of full model runs for display in PowerBI

All outputs and plots for all model runs are stored in AWS S3, and they are publicly available at [this website](http://www.autumn-data.com). Application _should_ be uploaded if the app crashes midway.

Each job is run on its own server, which is transient: it will be created for the job and will be destroyed at the end.

These AWS tasks are run by a command line tool which can be found in the `tasks` folder.

## Buildkite job runner

We have a self-serve job-runner website available [here](https://buildkite.com/autumn), build on the [Buildkite](https://buildkite.com/home) platform. This website can be used to run jobs in AWS. Buildkite runs on a small persistent server in AWS. Buildkite configuration is stored in `remote/buildkite/`.

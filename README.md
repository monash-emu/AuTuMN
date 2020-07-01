  
AuTuMN
======

![](https://github.com/monash-emu/AuTuMN/workflows/Continuous%20Integration/badge.svg)

This project is a Python disease modelling framework used by the [AuTuMN tuberculosis modelling project](http://www.tb-modelling.com/index.php).
It is currently being applied to COVID-19 as well. This project is used by the Monash Univeristy Epidemiological Modelling Unit.

See [this guide](./docs/setup.md) for information on how to set up this project. See [here](./docs/adding-new-covid-model.md) for steps on how to add a new COVID model.

## Using the command line

All of Autumn's features can be accessed from the command line. You can run commands as follows:

```bash
python -m apps <YOUR COMMANDS>
```

To see a list of options, try:

```bash
python -m apps --help
```

## Project structure

```
├── .github                 GitHub config
├── apps                    Specific applications of the framework
├── autumn                  AuTuMN framework module
├── data                    Data to be used by the models
├── docs                    Documentation and random files
├── scripts                 Utility scripts
├── summer                  SUMMER framework module
├── tests                   Automated tests
├── .gitignore              Files for Git to ignore
├── plots.py                Streamlit entrypoint
└── requirements.txt        Python library dependencies
```

## Running an application

You can run all the scenarios for specific application using the `run` command. For example, to run the "malaysia" region of the "covid" model, you can run:

```bash
python -m apps run covid malaysia
```

Model run outputs are written to `data/outputs/run` and can be viewed in Streamlit (see below).

## Running a calibration

You can run a model MCMC calibration as follows

```bash
python -m apps calibrate MODEL_NAME MAX_SECONDS RUN_ID
```

For example, to calibrate the malaysia COVID model for 30 seconds you can run:

```bash
python -m apps calibrate malaysia 30 0
```

The RUN_ID argument can always be "0" for local use, it doesn't really matter.

Model calibration outputs are written to `data/outputs/calibrate` and can be viewed in Streamlit (see below).

## Running Streamlit

We use [Streamlit](https://www.streamlit.io/) to visualise the output of local model runs. You can run streamlit from the command line to view your model's outputs as follows:

```bash
streamlit run plots.py
```

If you want to view the outputs of a calibration, run:

```bash
streamlit run plots.py mcmc
```

## Running the automated tests

We have a suite of automated tests that verify that the code works. Some of these are rigorous "unit" tests which validate functionality, while others are only "smoke" tests, which verify that the code runs without crashing. These tests are written with [pytest](https://docs.pytest.org/en/stable/).

You are encouraged to run the tests locally before pushing your code to GitHub. Automated tests may be run via [PyCharm](https://www.jetbrains.com/help/pycharm/pytest.html) or via the command line using pytest:

```
pytest -v
```

Tests are also run automatically via [GitHub Actions](https://github.com/monash-emu/AuTuMN/actions) on any pull request or commit to the `master` branch.

## Formatting

The codebase can be auto-formatted using [Black](https://github.com/psf/black):

```
./scripts/format.ps1
```

## Input data

Input data is stored in text format in the `data/inputs/` folder. All input data required to run the app should be stored in this folder, along with a README explaining its meaning and provenance. Input data is preprocessed into an SQLite database at runtime, inside the `autumn.inputs` module. A unique identified for the latest input data is stored in `data/inputs/input-hash.txt`. If you want to add new input data or modify existing data, then:

- add or update the source CSV/XLS files
- adjust the preprocess functions in `autumn.inputs` as required
- rebuild the database, forcing a new file hash to be written

To rebuild the database with new data, run:

```bash
python -m apps db build --force
```

## AWS calibration

We often need to run long, computationally expensive jobs. We are currently using Amazon Web Services (AWS) to do this. The scripts and documentation that allow you to do this can be found in the `scripts/aws/` folder. The following jobs are run in AWS:

- Calibration: Finding maximum likelihood parameters for some historical data using MCMC
- Full model runs: Running all scenarios for all accepted MCMC parameter sets
- PowerBI processing: Post-processing of full model runs for display in PowerBI

All outputs, logs and plots for all model runs are stored in AWS S3, and they are publicly available at [this website](http://autumn-data.s3-website-ap-southeast-2.amazonaws.com). Application _should_ be uploaded if the app crashes midway.

Each job is run on its own server, which is transient: it will be created for the job and will be destroyed at the end.

## Buildkite job runner

We have a self-serve job-runner website available [here](https://buildkite.com/autumn), build on the [Buildkite](https://buildkite.com/home) platform. This website can be used to run jobs in AWS. Buildkite runs on a small persistent server in AWS. Buildkite configuration is stored in `scripts/buildkite/`.

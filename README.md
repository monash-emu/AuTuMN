  
AuTuMN
======

![](https://github.com/monash-emu/AuTuMN/workflows/Continuous%20Integration/badge.svg)

This project is a modelling framework used by the [AuTuMN tuberculosis modelling project](http://www.tb-modelling.com/index.php). It provides a set of Python models that modularises the development of dynamic transmission models and allows a pluggable API to other system. Applied to tuberculosis.

The tuberculosis-specific AuTuMN modelling framework is build on top of the disease-agnostic [SUMMER project](https://github.com/monash-emu/summer).

See [this guide](./docs/setup.md) for information on how to set up this project.

## Project Structure

```
├── .github                 GitHub config
├── apps            Specific apps of AuTuMN
├── autumn                  AuTuMN framework module
├── data                    Data to be used by the models
├── docs                    Documentation
├── scripts                 Ad-hoc utility scripts
├── tests                   Automated tests
├── .gitignore              Files for Git to ignore
├── .pylintrc               PyLint code linter configuration
├── requirements.txt        Python library dependencies
└── setup.py                Packaging for deployment to MASSIVE computing platform
```

## MASSIVE

We sometimes need to run calibration jobs on Monash's [MASSIVE](https://www.monash.edu/research/infrastructure/platforms-pages/massive) computer cluster. The scripts and documentation that allow you to do this can be found in the `scripts/massive/` folder.

## AWS Calibration

We sometimes need to run jobs using Amazon Web Services. The scripts and documentation that allow you to do this can be found in the `scripts/aws/` folder.

## Tests

Automated tests may be run via [PyCharm](https://www.jetbrains.com/help/pycharm/pytest.html) or via the command line:

```
./scripts/test.ps1
```

Tests are also run automatically via [GitHub Actions](https://github.com/monash-emu/AuTuMN/actions) on any pull request or commit to the `master` branch.

## Formatting

The codebase can be auto-formatted using [Black](https://github.com/psf/black):

```
./scripts/format.ps1
```

## Running apps

Specific uses of the AuTuMN framework are present in `apps/`. You can run an application through an IDE like PyCharm, or run it from the command line:

```
./scripts/run.ps1 --help
```

# Anaconda commands

*work in progress 22/11/2020*

See conda env docs 
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Create a new environment

```bash
conda create --prefix ./condaenv
```

### Activate an environment

Activate/deactivate an existing environment

```bash
conda activate ./condaenv
conda deactivate
```

### Updating an environment

Update an existing environment to use the latest packages.

```bash
conda env update --prefix ./condaenv --file conda.yml --prune
```

To add a new package, add it to `conda.yml` then run the update.
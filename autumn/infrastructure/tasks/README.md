### Remote tasks

Runs AuTuMN tasks, which are large jobs to be run on remote servers.
This module requires AWS access to run.

You can access these tasks via the CLI:

```
python -m autumn tasks
```

### Run a calibration

```bash
RUN_ID="covid_19/malaysia/111111111/bbbbbbb"
RUN_ID="covid_19/philippines/111111111/bbbbbbb"
RUN_ID="tuberculosis/marshall-islands/111111111/aaaaaaa"
```

# Run a calibration

python -m autumn tasks calibrate --run $RUN_ID --chains 1 --runtime 120 --verbose

# Run full models

python -m autumn tasks full --run $RUN_ID --burn 0 --sample 12 --verbose

# Run PowerBI processing

python -m autumn tasks powerbi --run $RUN_ID

````

### Clean up test task

Clean up everything

```bash
rm -rf data/outputs/remote
aws --profile autumn s3 rm --quiet --recursive "s3://autumn-data/covid_19/manila/111111111/aaaaaaa"
aws --profile autumn s3 rm --quiet --recursive "s3://autumn-data/tuberculosis/marshall-islands/111111111/aaaaaaa"
````

### Rebuild the website

The test run data can be viewed at

- http://www.autumn-data.com/app/covid_19/region/malaysia/run/111111111-bbbbbbb.html
- http://www.autumn-data.com/app/covid_19/region/marshall-islands/run/111111111-aaaaaaa.html

But you may need to rebuild the website first:

```bash
./scripts/website/deploy.sh
```

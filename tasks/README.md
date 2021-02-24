### Luigi tasks

Runs AuTuMN tasks, which are large jobs to be run on remote servers.
This module requires AWS access to run.

You can access these tasks via the CLI:

```
python -m tasks
```

### Run a calibration

```bash
RUN_ID="covid_19/malaysia/111111111/aaaaaaa"
RUN_ID="tuberculosis/marshall-islands/111111111/aaaaaaa"


# Run a calibration
python -m tasks calibrate --run  $RUN_ID --chains 1 --runtime 30 --verbose

# Run full models
python -m tasks full --run  $RUN_ID --burn 1 --verbose

# Run PowerBI processing
python -m tasks powerbi --run $RUN_ID

# Run DHHS processing
python -m tasks dhhs --commit bbbbbbb
```

### Clean up test task

Clean up everything

```bash
rm -rf data/outputs/remote
aws --profile autumn s3 rm --quiet --recursive "s3://autumn-data/covid_19/manila/111111111/aaaaaaa"
aws --profile autumn s3 rm --quiet --recursive "s3://autumn-data/tuberculosis/marshall-islands/111111111/aaaaaaa"
```

### Rebuild the website

The test run data can be viewed at

- http://www.autumn-data.com/app/covid_19/region/manila/run/111111111-aaaaaaa.html
- http://www.autumn-data.com/app/covid_19/region/marshall-islands/run/111111111-aaaaaaa.html

But you may need to rebuild the website first:

```bash
./scripts/website/deploy.sh
```

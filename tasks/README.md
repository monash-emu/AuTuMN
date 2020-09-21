### Luigi tasks

Runs AuTuMN tasks with Luigi, which are large jobs to be run on remote servers.
This module requires AWS access to run.

You can access these tasks via the CLI:

```
python -m tasks
```

### Run a calibration

```
RUN_ID="covid_19/manila/111111111/aaaaaaa"
RUN_ID="tuberculosis/marshall-islands/111111111/aaaaaaa"

python -m tasks calibrate --run  $RUN_ID --chains 4 --runtime 60 --workers 4
```


### Run full models

```
python -m tasks full --run  $RUN_ID --burn 1 --workers 4
```

### Run PowerBI processing

```
python -m tasks powerbi --run $RUN_ID --workers 4
```

### Run DHHS processing

```
python -m tasks dhhs --commit aaaaaaa --workers 7
```

### Clean up test task

Clean up everything. Always type in the run id manually or you might accidentally delete *everything*.

```bash
rm -rf data/outputs/remote data/outputs/calibrate
aws --profile autumn s3 rm --quiet --recursive s3://autumn-data/covid_19/manila/111111111/aaaaaaa
aws --profile autumn s3 rm --quiet --recursive s3://autumn-data/tuberculosis/marshall-islands/111111111/aaaaaaa
```


### Rebuild the website

To rebuild the website at http://www.autumn-data.com/app/covid_19/region/manila/run/111111111-aaaaaaa.html

```bash
./scripts/website/deploy.sh
```


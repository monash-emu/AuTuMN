### Luigi tasks

Runs AuTuMN tasks with Luigi, which are large jobs to be run on remote servers.
This module requires AWS access to run.

You can access these tasks via the CLI:

```
python -m tasks
```

### Run a calibration

```
python -m tasks calibrate --run  manila-111111111-aaaaaaa --chains 4 --runtime 60 --workers 4
```


### Run full models

```
python -m tasks full --run  manila-111111111-aaaaaaa --burn 2 --workers 4
```

### Run PowerBI processing

```
python -m tasks powerbi --run  manila-111111111-aaaaaaa --workers 4
python -m tasks powerbi --run  philippines-1599783984-46edf4d --workers 1
```

### Run DHHS processing

```
python -m tasks dhhs --commit aaaaaaa --workers 7
```

### Clean up test task

```bash
rm -rf data/outputs/remote data/outputs/calibrate
aws --profile autumn s3 rm --quiet --recursive s3://autumn-data/manila-111111111-aaaaaaa
```

### Rebuild the website

To rebuild the website at http://www.autumn-data.com/model/manila/run/manila-111111111-aaaaaaa.html

```bash
./scripts/website/deploy.sh
```


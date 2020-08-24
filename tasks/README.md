### Luigi tasks

Runs AuTuMN tasks with Luigi, which are large jobs to be run on remote servers.
This module requires AWS access to run.

You can access these tasks via the CLI:

```
python -m tasks
```

### Run a calibration

```
python -m tasks calibrate --run  manila-111111111-aaaaaaa --chains 2 --runtime 30 --workers 2
```

### Run full models

```
python -m tasks full --run  manila-111111111-aaaaaaa --burn 0 --workers 2
```

### Run PowerBI processing

```
python -m tasks powerbi --run  manila-111111111-aaaaaaa --workers 2
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


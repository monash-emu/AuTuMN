# MASSIVE Job Scripts

This folder contains scripts which help you run parameter jobs on the [MASSIVE](https://www.monash.edu/research/infrastructure/platforms-pages/massive) computer cluster.

The login node is at m3.massive.org.au. Our project folder is `/project/sh30`.

# Running a MASSIVE job

Follow the following steps to run a job.

### Modify config

Modify your code to match how you want it to run, and modify the config file at `scripts/massive/config.yml`, then commit the changes to Git, and push them up to GitHub.

Notably, you want to set up your `job_command` to execute the Python script that you need to run. For example, to run the Mongolia calibration:

```yaml
job_command: python3 -m applications mongolia-calibration 10000 30
```

Note that this command is always called with an extra argument, which is the Slurm array task ID, so when the script runs it will actually be something like:

```bash
SLURM_ARRAY_TASK_ID=5
python3 -m applications mongolia-calibration 300 $SLURM_ARRAY_TASK_ID
```

Please use `python3` instead of `python` when writing this command.

### Setup the job

Use SSH to access the M3 login server using your M3 username and password. Eg. as Romain, you would use:

```bash
ssh rrag0004@m3.massive.org.au
```

Navigate to the AuTuMN repo in our project folder, and make sure to pull down the latest changes from GitHub. Make sure you are on the right branch and commit.

```bash
cd /projects/sh30/autumn-repo
git pull  # Pull the latest code
git branch  # Check the branch
git log -1  # Check the commit
```

From the AuTuMN repo folder, run the setup script with your job name. For example:

```bash
./scripts/massive/setup.sh my-job
```

This will create a folder in `/projects/sh30/autumn-jobs` for your job, eg: `/projects/sh30/autumn-repo/my-job-17-02-2020`.

### Run the job

Navigate to the job folder and run the job:

```bash
cd /projects/sh30/autumn-repo/my-job-17-02-2020
./run-job.sh
```

After that the job will be submitted to Slurm. You can view the job's progress with:

```bash
squeue -j <JOBID> # or
squeue -u <USERNAME>
```

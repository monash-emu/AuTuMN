# Adding a new COVID model

How to add a new COVID model

- Add your region's name to `autumn.constants.Region`
- Add your region's default parameters to `apps/covid_19/params/{region_name}/default.yml` (see other similar files for an example)
- Add your region's plotting config to `apps/covid_19/plots/{region_name}.yml` (see other similar files for an example)
- Create and register a RegionApp for your region in `apps.covid_19.__init__`

You can now run your model from the command line

```bash
python -m apps run region-name
```

# Registering a COVID model for calibration

- Add your calibration script to `apps.covid_19.calibration` under your region's name (snake case)
- Update `apps.covid_19.calibration.__init__` to register your calibration script

You can now run a calibration

```bash
python -m apps calibrate region-name
```

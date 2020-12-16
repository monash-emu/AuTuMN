# Summer documentation

This folder user Sphinx to automatically build reference documentation for the summer library.
The documentation is automatically built and deployed to [summerepi.com](http://summerepi.com/) whenever code is pushed to `master`.

To install requirements

```bash
pip install -r docs/requirements.txt
```

To build and deploy

```bash
./scripts/docs/build.sh
./scripts/docs/deploy.sh
```

To work on docs locally

```bash
pip install watchdog[watchmedo]
./scripts/docs/watch.sh
# In a separate terminal
./scripts/docs/serve.sh
```

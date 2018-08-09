import os
import shutil

this_dir = os.path.dirname(__file__)
config_fname = os.path.join(this_dir, 'server/config.py')
default_config_fname = os.path.join(this_dir, 'server/config_default.py')
if not os.path.isfile(config_fname):
    shutil.copy(default_config_fname, config_fname)

from server.api import app
from server import config

if __name__ == "__main__":
    app.run(
        threaded=True,
        use_reloader=True,
        port=int(config.PORT))

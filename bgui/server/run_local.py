import os
import shutil

if not os.path.isfile("server/config.py"):
    shutil.copy("server/config_default.py", "server/config.py")

from server.api import app
from server import config

app.run(
    threaded=True,
    use_reloader=True,
    debug=True,
    use_debugger=False,
    port=int(config.PORT))

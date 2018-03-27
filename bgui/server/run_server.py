import os
import sys
import shutil

# Auto-generates config.py if not found
if not os.path.isfile("config.py"):
    shutil.copy("config_default.py", "config.py")

# Hack to load sibling modules in "__main__" script
try:
    import server
except:
    sys.path.insert(0, os.path.abspath(".."))
from server.config import PORT
from server.api import app

app.run(
    threaded=True,
    use_reloader=True,
    debug=True,
    use_debugger=False,
    port=int(PORT))

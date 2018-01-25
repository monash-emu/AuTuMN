import sys
import os
import shutil

# Must be run in the server directory

# Auto-generates config.py if not found
if not os.path.isfile("config.py"):
    shutil.copy("config_default.py", "config.py")

# Because we are in a non-package, need to use
# this trick to include the current directory
# in the module search path
try:
    import server
except:
    sys.path.insert(0, os.path.abspath(".."))

from server import _autoreload
from server import _twisted_wsgi

_autoreload.main(_twisted_wsgi.run)


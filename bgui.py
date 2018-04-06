import os
import sys
from subprocess import call
os.chdir('bgui/server')
call([sys.executable, "run_client_server.py"])


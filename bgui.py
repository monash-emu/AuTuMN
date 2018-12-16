import os
import sys
from subprocess import call
print(os.getcwd())
run_server_path = os.path.join(os.getcwd() + r'\bgui\server')
os.chdir(run_server_path)
print(os.getcwd())
call([sys.executable, "run_client_server.py"])


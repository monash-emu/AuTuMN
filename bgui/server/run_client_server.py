from __future__ import print_function
import os
import sys
import shutil
import threading
import time
import urllib
import webbrowser
import subprocess

if not os.path.isfile("server/config.py"):
    shutil.copy("server/config_default.py", "server/config.py")
from server import config

# creates a thread to poll server before opening client
url = 'http://localhost:' + str(config.PORT)
i = 0
def open_page_delayed():
    is_not_loaded = True
    while is_not_loaded:
        time.sleep(1)
        print("Waiting", i, "s for", url)
        try:
            response_code = urllib.urlopen(url).getcode()
            is_not_loaded = response_code >= 400
        except:
            is_not_loaded = True
    print("Opening", url)
    webbrowser.open(url)
threading.Thread(target=open_page_delayed).start()

# creates new process for flask in threaded debug mode
# so that it doesn't interfere with the thread
# for the web-browser polling
subprocess.call([sys.executable, "run_local.py"])
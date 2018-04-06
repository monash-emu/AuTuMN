from __future__ import print_function
import os
import sys
import shutil
import threading
import time
import urllib
import webbrowser
import subprocess


# creates a thread to poll server before opening client
def open_page_delayed():
    is_not_loaded = True
    i = 0
    while is_not_loaded:
        try:
            from server import config
            url = 'http://localhost:' + str(config.PORT)
            response_code = urllib.urlopen(url).getcode()
            is_not_loaded = response_code >= 400
        except:
            time.sleep(1)
            i += 1
            print("Waiting", i, "s for server")
            is_not_loaded = True
    print("Opening", url)
    webbrowser.open(url)
threading.Thread(target=open_page_delayed).start()

# creates new process for flask in threaded debug mode
# so that it doesn't interfere with the thread
# for the web-browser polling
subprocess.call([sys.executable, "run_local.py"])
from __future__ import print_function
import subprocess
import os
import webbrowser
import threading
import time
import urllib

url = 'http://localhost:3000'

def open_page_delayed():
    not_loaded = True
    while not_loaded:
        time.sleep(1)
        try:
            response_code = urllib.urlopen(url).getcode()
            not_loaded = response_code >= 400
        except:
            not_loaded = True
    print("Opening webclient at", url)
    webbrowser.open(url)
threading.Thread(target=open_page_delayed).start()

os.chdir('bgui/server')
subprocess.call(["python", "run_server.py"])


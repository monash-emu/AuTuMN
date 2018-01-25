import subprocess
import os
import webbrowser
import threading
import time

def open_page_delayed():
    time.sleep(5)
    webbrowser.open('http://localhost:3000')

threading.Thread(target=open_page_delayed).start()

os.chdir('bgui/server')
subprocess.call(["python", "run_server.py"])


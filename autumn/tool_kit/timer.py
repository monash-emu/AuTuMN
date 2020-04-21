from time import time


class Timer:
    """
    Prints the times that a block of code takes to run
    """

    def __init__(self, message):
        self.message = message
        self.start = None

    def __enter__(self):
        self.start = time()
        msg = self.message[0].upper() + self.message[1:]
        print(f"{msg}...")

    def __exit__(self, *args):
        runtime = time() - self.start
        msg = self.message[0].lower() + self.message[1:]
        print(f"Finished {msg} in {runtime:0.1f} seconds.")

# src/utils.py

import sys
import os

class Logger:
    """Redirects stdout to both terminal and a log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")
        self.encoding = "utf-8"   # ✅ Fix: mimic real stdout encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False   # ✅ Some libs check this

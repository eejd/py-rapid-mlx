#!/usr/bin/env python3
"""Give a sandboxed helper its own process group before exec."""
import os
import sys

if os.getpgrp() != os.getpid():
    os.setsid()
os.execv("/usr/bin/sandbox-exec", ["/usr/bin/sandbox-exec", *sys.argv[1:]])

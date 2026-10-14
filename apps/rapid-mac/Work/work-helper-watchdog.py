#!/usr/bin/env python3
"""Run a Work helper in its own process group and reap it if Rapid exits."""

import os
import signal
import sys
import time

owner = int(sys.argv[1])
command = sys.argv[2:]
try:
    os.setsid()
except PermissionError:
    # Foundation's Process may have made this child a group leader already.
    if os.getpgrp() != os.getpid():
        raise
group = os.getpgrp()
if os.fork() == 0:
    # The fork stays alive when exec replaces the helper launcher.
    # Stay alive for the SIGKILL escalation after signalling our own group.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.2)
        try:
            os.kill(owner, 0)
        except ProcessLookupError:
            os.killpg(group, signal.SIGTERM)
            time.sleep(0.2)
            os.killpg(group, signal.SIGKILL)
            os._exit(0)
else:
    os.execv(command[0], command)

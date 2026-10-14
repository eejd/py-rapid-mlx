"""Regression check for Work helpers surviving their owning app."""

import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

WATCHDOG = pathlib.Path(__file__).with_name("work-helper-watchdog.py")
OWNER_SCRIPT = """
import os
import pathlib
import subprocess
import sys
import time

watchdog, pid_path, ready_path, group_leader, resistant = sys.argv[1:]
helper_code = (
    "import pathlib,signal,sys,time; "
    + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if resistant == "1" else "")
    + "pathlib.Path(sys.argv[1]).touch(); time.sleep(30)"
)
helper = subprocess.Popen(
    [
        sys.executable, "-I", "-S", watchdog, str(os.getpid()), sys.executable, "-c",
        helper_code,
        ready_path,
    ],
    start_new_session=group_leader == "1",
)
pathlib.Path(pid_path).write_text(str(helper.pid))
ready = pathlib.Path(ready_path)
deadline = time.monotonic() + 3
while not ready.exists() and time.monotonic() < deadline:
    if helper.poll() is not None:
        sys.exit(2)
    time.sleep(0.05)
sys.exit(0 if ready.exists() else 3)
"""


class WatchdogTests(unittest.TestCase):
    def test_helper_exits_when_owner_disappears(self):
        for starts_as_group_leader, resistant in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with (
                self.subTest(
                    starts_as_group_leader=starts_as_group_leader, resistant=resistant
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                pid_file = pathlib.Path(directory) / "helper.pid"
                ready_file = pathlib.Path(directory) / "ready"
                owner = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        OWNER_SCRIPT,
                        str(WATCHDOG),
                        str(pid_file),
                        str(ready_file),
                        "1" if starts_as_group_leader else "0",
                        "1" if resistant else "0",
                    ],
                    check=True,
                )
                self.assertEqual(owner.returncode, 0)
                pid = int(pid_file.read_text())
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                else:
                    os.kill(pid, 9)
                    self.fail(
                        "Work helper survived its owning process for five seconds"
                    )


if __name__ == "__main__":
    unittest.main()

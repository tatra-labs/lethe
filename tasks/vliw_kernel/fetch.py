"""Download Anthropic's original performance take-home at a pinned commit.

It is not redistributed with lethe, and its terms ask that solutions not be published:
keep the runs you make on this task to yourself.
"""

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "5452f74bd977807ac2e74f3d29432b9df6f25197"
URL = f"https://github.com/anthropics/original_performance_takehome/archive/{COMMIT}.zip"

here = Path(__file__).resolve().parent
with urllib.request.urlopen(URL, timeout=120) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))
prefix = f"original_performance_takehome-{COMMIT}/"

staging = here / "candidate.partial"
shutil.rmtree(staging, ignore_errors=True)
staging.mkdir()
for name in ("perf_takehome.py", "problem.py"):
    (staging / name).write_bytes(archive.read(prefix + name))
(here / "judge" / "frozen_problem.py").write_bytes(archive.read(prefix + "tests/frozen_problem.py"))
staging.rename(here / "candidate")
print(f"fetched the take-home at {COMMIT[:7]} into {here / 'candidate'}")

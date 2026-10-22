import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from value import X  # noqa: E402

print(json.dumps({"score": X, "feedback": f"X is {X}"}))

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench.gpu_lock import gpu_lock

target = sys.argv[1]

with gpu_lock(purpose=f"A-06 premise probe: {target}", timeout_s=900):
    mod = __import__(target)
    mod.main()

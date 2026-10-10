"""Phase 1, test 1: does a Docker --memory cap stop a container from taking GPU/unified memory on GB10?

Run INSIDE the capped container (docker run --rm --gpus all --memory=7g --memory-swap=7g ... python
scripts/phase1_memcap.py). It allocates CUDA memory in 512 MB steps until it fails, printing each step.
Pass = it dies or errors well before host MemAvailable drops into GLM's territory. Run scripts/glm_health.sh
in another terminal the whole time.
"""
import sys
import time

import torch

STEP_MB = 512
LIMIT_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 10240  # never ask for more than this, whatever the cap says
blocks = []
for i in range(LIMIT_MB // STEP_MB):
    try:
        blocks.append(torch.empty(STEP_MB * 1024 * 1024, dtype=torch.uint8, device="cuda"))
        blocks[-1].fill_(1)  # touch it so it is really resident
    except RuntimeError as e:
        print(f"allocation failed at {len(blocks) * STEP_MB} MB: {e}")
        break
    with open("/proc/meminfo") as f:
        avail = next(l for l in f if l.startswith("MemAvailable:")).split()[1]
    print(f"{(i + 1) * STEP_MB:6d} MB allocated, host MemAvailable {int(avail) // 1024} MB", flush=True)
    time.sleep(0.5)
else:
    print(f"reached the {LIMIT_MB} MB self-limit without the cgroup cap intervening: the cap does NOT cover GPU memory")

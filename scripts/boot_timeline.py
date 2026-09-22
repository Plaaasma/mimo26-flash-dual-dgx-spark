#!/usr/bin/env python3
"""boot_timeline.py [container]: per-stage timing of the last boot from the head container's timestamped log."""
import datetime as dt, re, subprocess, sys
c = sys.argv[1] if len(sys.argv) > 1 else "mimo26-head"
lines = subprocess.run(["docker", "logs", "--timestamps", c], capture_output=True, text=True).stdout.splitlines()
lines += subprocess.run(["docker", "logs", "--timestamps", c], capture_output=True, text=True).stderr.splitlines()
lines.sort()
def ts(l):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z", l)
    return dt.datetime.fromisoformat(m.group(1)[:26]).timestamp() if m else None
t0 = next((ts(l) for l in lines if ts(l)), None)
marks = [
    ("patches applied",     r"launching: vllm serve"),
    ("engine core init",    r"Initializing a V1 LLM engine"),
    ("kernels chosen",      r"Mxfp4 MoE backend"),
    ("weights start",       r"checkpoint shards:\s+0%|using InstantTensor loader:\s+0%|Loading safetensors using InstantTensor"),
    ("weights done",        r"Loading weights took"),
    ("model loading done",  r"Model loading took"),
    ("drafter loaded",      "SECOND:Loading weights took"),
    ("compile done",        r"Compiling a graph .* takes|torch.compile .* took|Directly load the compiled graph"),
    ("KV cache sized",      r"GPU KV cache size"),
    ("graph capture done",  r"Graph capturing finished|Capturing CUDA graphs .*finished|CUDA graphs captured"),
    ("API ready",           r"Application startup complete"),
    ("container exited",    "LAST"),
]
prev = t0
print(f"{'stage':22s} {'at +s':>8s} {'delta s':>8s}")
for name, rx in marks:
    t = None
    if rx.startswith("SECOND:"):
        hits = [ts(l) for l in lines if rx[7:] in l]
        t = hits[1] if len(hits) > 1 else None
    elif rx == "LAST":
        t = ts(lines[-1]) if lines else None
    else:
        for l in lines:
            if re.search(rx, l):
                t = ts(l); break
    if t is None:
        print(f"{name:22s} {'-':>8s}"); continue
    print(f"{name:22s} {t - t0:8.1f} {t - prev:8.1f}"); prev = t

#!/usr/bin/env python3
"""prof_summary.py <trace.json[.gz]> [--top N]: summarize a vLLM torch-profiler trace: total CUDA kernel time by
kernel name (top N), CPU-side op time, number of engine steps and mean step wall time (from the CPU 'execute_model'
or forward markers), plus the largest gaps where the GPU sits idle. Lets us see where a decode step goes."""
import sys, json, gzip, re, collections
path = sys.argv[1]; top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data
kern = collections.defaultdict(lambda: [0.0, 0]); cpu = collections.defaultdict(lambda: [0.0, 0])
kernel_spans = []; step_marks = []
for e in ev:
    if e.get("ph") != "X": continue
    cat = e.get("cat", ""); name = e.get("name", ""); dur = e.get("dur", 0.0); ts = e.get("ts", 0.0)
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        kern[name][0] += dur; kern[name][1] += 1; kernel_spans.append((ts, ts + dur))
    elif cat in ("cpu_op", "user_annotation", "python_function"):
        cpu[name][0] += dur; cpu[name][1] += 1
        if re.search(r"execute_model|forward|model_runner|sample", name) and cat == "user_annotation":
            step_marks.append((name, ts, dur))
tot_k = sum(v[0] for v in kern.values())
print(f"CUDA kernel time total: {tot_k / 1e3:.1f} ms over {sum(v[1] for v in kern.values())} launches")
if kernel_spans:
    kernel_spans.sort(); span = kernel_spans[-1][1] - kernel_spans[0][0]
    busy = 0.0; cur_s, cur_e = kernel_spans[0]
    for s, e in kernel_spans[1:]:
        if s > cur_e: busy += cur_e - cur_s; cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    print(f"GPU span {span / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms ({busy / span:.0%}), idle {(span - busy) / 1e3:.1f} ms")
print(f"\n{'CUDA kernel':90s} {'total ms':>9s} {'calls':>6s} {'us/call':>8s} {'share':>6s}")
for name, (t, n) in sorted(kern.items(), key=lambda x: -x[1][0])[:top]:
    print(f"{name[:90]:90s} {t / 1e3:9.2f} {n:6d} {t / n:8.1f} {t / tot_k:6.1%}")
ann = collections.defaultdict(lambda: [0.0, 0])
for name, ts, dur in step_marks: ann[name][0] += dur; ann[name][1] += 1
if ann:
    print("\nannotations:")
    for name, (t, n) in sorted(ann.items(), key=lambda x: -x[1][0])[:12]: print(f"  {name[:70]:70s} {t / 1e3:9.2f} ms {n:6d} calls {t / n / 1e3:8.2f} ms/call")

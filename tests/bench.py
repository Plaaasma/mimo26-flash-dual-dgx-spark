#!/usr/bin/env python3
"""bench.py: decode throughput at N concurrent streams and cold-prefill speed through the OpenAI API.

  tests/bench.py --url http://127.0.0.1:8888 --model mimo-v2.6-flash --conc 1 6 --out 400
  tests/bench.py ... --prefill 2000 32000 128000      (unique prompts of that many tokens, one at a time)

Only run on an idle server (it saturates the engine). Reports aggregate and per-stream tok/s from the usage
fields, TTFT from the first streamed chunk, and prefill tok/s = prompt_tokens / TTFT.
"""
import argparse, json, random, statistics, threading, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument("--url", default="http://127.0.0.1:8888"); P.add_argument("--model", default="mimo-v2.6-flash")
P.add_argument("--key", default=""); P.add_argument("--conc", type=int, nargs="*", default=[1, 6])
P.add_argument("--out", type=int, default=400); P.add_argument("--prefill", type=int, nargs="*", default=[])
P.add_argument("--thinking", default="false")
A = P.parse_args()
H = {"Content-Type": "application/json"}
if A.key: H["Authorization"] = f"Bearer {A.key}"
TASKS = ["Write a Python function that parses ISO-8601 timestamps and explain edge cases.",
         "Explain how a B-tree differs from an LSM tree for write-heavy workloads.",
         "Write a bash script that rotates logs older than 7 days and compresses them.",
         "Describe the TCP three-way handshake and what happens on packet loss.",
         "Write a short essay on why unified memory changes inference deployment.",
         "Implement binary search in Rust with tests.", "List 20 facts about the Moon.",
         "Write a JSON schema for a customer order with nested line items."]

def stream(prompt, max_tokens):
    body = {"model": A.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": A.thinking == "true"}}
    req = urllib.request.Request(f"{A.url}/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for ln in r:
            if not ln.startswith(b"data:"): continue
            d = ln[5:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            if ttft is None and j.get("choices") and (j["choices"][0].get("delta") or {}).get("content"): ttft = time.time() - t0
            if j.get("usage"): usage = j["usage"]
    return {"ttft": ttft, "wall": time.time() - t0, "usage": usage or {}}

def decode_bench(conc):
    res = [None] * conc
    def w(i):
        salt = f"[run {random.randrange(1 << 30)}] "   # unique prefix: no prefix-cache hits
        res[i] = stream(salt + TASKS[i % len(TASKS)], A.out)
    ths = [threading.Thread(target=w, args=(i,)) for i in range(conc)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
    out = [r["usage"].get("completion_tokens", 0) for r in res]
    per = [o / r["wall"] for o, r in zip(out, res) if r["wall"] > 0]
    ttfts = [r["ttft"] for r in res if r["ttft"]]
    print(f"C{conc}: aggregate {sum(out) / wall:6.1f} tok/s | per-stream {statistics.mean(per):5.1f} tok/s | "
          f"TTFT mean {statistics.mean(ttfts) if ttfts else 0:.2f}s | {sum(out)} tokens in {wall:.1f}s")

def prefill_bench(n_tokens):
    words = ["alpha", "bridge", "carbon", "delta", "ember", "falcon", "granite", "harbor", "iris", "jade", "kernel", "lumen"]
    rnd = random.Random(n_tokens * 7919 + random.randrange(1 << 20))
    text = " ".join(rnd.choice(words) + str(rnd.randrange(1000)) for _ in range(int(n_tokens / 1.35)))
    r = stream("Here is a document:\n" + text + "\n\nReply with the single word: done", 8)
    pt = r["usage"].get("prompt_tokens", 0); ttft = r["ttft"] or r["wall"]
    print(f"prefill {pt:7d} tokens: TTFT {ttft:7.2f}s = {pt / ttft:7.1f} tok/s")

for n in A.prefill: prefill_bench(n)
for c in A.conc: decode_bench(c)

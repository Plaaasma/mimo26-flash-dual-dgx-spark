#!/usr/bin/env python3
"""long_ctx.py: long-context needle + timing through the OpenAI API.

  tests/long_ctx.py --url http://HEAD:8888 --tokens 128000 256000 500000 [--depth 0.1 0.5 0.9]

For each size: a unique filler document (no prefix-cache hits) with a random 6-digit passcode planted at each depth;
asks for the passcode. Streams the reply and reports: client->server request bytes, time to first token, prefill
tok/s (prompt tokens / TTFT), whether the answer is right, plus the server's own per-request timings from /metrics
deltas (queue time, prefill time, inference time) so input-processing overhead = TTFT - (queue + prefill) shows up.
"""
import argparse, json, random, re, time, urllib.request

P = argparse.ArgumentParser()
P.add_argument("--url", default="http://127.0.0.1:8888"); P.add_argument("--model", default="mimo-v2.6-flash")
P.add_argument("--tokens", type=int, nargs="+", default=[128000]); P.add_argument("--depth", type=float, nargs="+", default=[0.5])
P.add_argument("--thinking", default="false")
A = P.parse_args()
H = {"Content-Type": "application/json"}
WORDS = ("amber basin cobalt drift ember fjord granite harbor island juniper kestrel lagoon meadow nectar orchid "
         "prairie quartz ridge summit tundra umber valley willow xenon yarrow zephyr").split()
TOK_PER_WORD = 1.25   # measured roughly for this filler with MiMo's tokenizer; the server's usage block is the truth


def metrics():
    t = urllib.request.urlopen(A.url + "/metrics", timeout=10).read().decode()
    out = {}
    for name in ("request_queue_time_seconds", "request_prefill_time_seconds", "request_inference_time_seconds",
                 "time_to_first_token_seconds", "e2e_request_latency_seconds"):
        m = re.search(r"^vllm:" + name + r"_sum\{[^}]*\} ([0-9.e+-]+)", t, re.M)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def run(n_tokens, depth):
    rnd = random.Random(n_tokens * 1009 + int(depth * 100) + random.randrange(1 << 30))
    code = f"{rnd.randrange(10**5, 10**6)}"
    n_words = int(n_tokens / TOK_PER_WORD)
    words = [rnd.choice(WORDS) for _ in range(n_words)]
    pos = int(depth * n_words)
    doc = " ".join(words[:pos]) + f"\n\nThe secret passcode is {code}. Remember it.\n\n" + " ".join(words[pos:])
    body = {"model": A.model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 32, "temperature": 0,
            "messages": [{"role": "user", "content": "Read this document carefully.\n\n" + doc +
                          "\n\nWhat is the secret passcode mentioned in the document? Reply with the number only."}],
            "chat_template_kwargs": {"enable_thinking": A.thinking == "true"}}
    data = json.dumps(body).encode()
    m0 = metrics(); t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", data=data, headers=H), timeout=7200) as r:
        for ln in r:
            if not ln.startswith(b"data:"): continue
            d = ln[5:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            if j.get("choices"):
                delta = j["choices"][0].get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece and ttft is None: ttft = time.time() - t0
                text += delta.get("content") or ""
            if j.get("usage"): usage = j["usage"]
    wall = time.time() - t0; m1 = metrics()
    d = {k: m1[k] - m0[k] for k in m0}
    pt = usage.get("prompt_tokens", 0); ttft = ttft or wall
    overhead = ttft - d["request_queue_time_seconds"] - d["request_prefill_time_seconds"]
    ok = code in text
    print(f"{pt:8d} tok depth {depth:.1f}: {'OK ' if ok else 'MISS'} ({text.strip()[:24]!r} vs {code}) | req {len(data)/1e6:6.1f} MB | "
          f"TTFT {ttft:7.1f}s = {pt / ttft:7.1f} tok/s | server: queue {d['request_queue_time_seconds']:6.2f}s "
          f"prefill {d['request_prefill_time_seconds']:7.1f}s | outside engine {overhead:6.1f}s", flush=True)


for n in A.tokens:
    for dp in A.depth:
        run(n, dp)

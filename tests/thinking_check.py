#!/usr/bin/env python3
"""thinking_check.py: every way a client can switch MiMo's thinking on or off, streamed and not, through the
OpenAI API. Checks per case: reasoning present or absent as requested, the answer in `content`, no reasoning or
<think> tags leaking into content, tool calls parsed, multi-turn with the previous reasoning passed back.

  tests/thinking_check.py --url http://HEAD:8888 [--model mimo-v2.6-flash]
"""
import argparse, json, urllib.request

P = argparse.ArgumentParser(); P.add_argument("--url", default="http://127.0.0.1:8888"); P.add_argument("--model", default="mimo-v2.6-flash")
A = P.parse_args()
H = {"Content-Type": "application/json"}
Q = "What is 17*23? Answer with the number only."
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
fails = 0


def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H), timeout=900))


def stream(body):
    body = {**body, "stream": True}
    reasoning = content = ""; tool_calls = {}
    with urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H), timeout=900) as r:
        for ln in r:
            if not ln.startswith(b"data:"): continue
            d = ln[5:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            for ch in j.get("choices") or []:
                dl = ch.get("delta") or {}
                reasoning += dl.get("reasoning_content") or dl.get("reasoning") or ""
                content += dl.get("content") or ""
                for tc in dl.get("tool_calls") or []:
                    t = tool_calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    t["name"] += f.get("name") or ""; t["arguments"] += f.get("arguments") or ""
    return reasoning, content, [(t["name"], t["arguments"]) for t in tool_calls.values()]


def msg(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or ""), (m.get("content") or ""), \
        [(t["function"]["name"], t["function"]["arguments"]) for t in (m.get("tool_calls") or [])]


def check(label, reasoning, content, want_reasoning, want_in_content=None, tools=None, want_tool=None):
    global fails
    leak = any(t in content for t in ("<think>", "</think>"))
    ok = (bool(reasoning.strip()) == want_reasoning) and not leak
    if want_in_content is not None: ok &= want_in_content in content
    if want_tool is not None: ok &= any(n == want_tool for n, _ in (tools or []))
    fails += 0 if ok else 1
    print(f"{'OK  ' if ok else 'FAIL'} {label:48s} reasoning {len(reasoning):5d} chars | content {content.strip()[:60]!r}"
          + (f" | tools {tools}" if tools else "") + (" | <think> LEAK" if leak else ""))


base = {"model": A.model, "messages": [{"role": "user", "content": Q}], "max_tokens": 2048, "temperature": 0}
cases = [
    ("server default (no parameter)", {}, None),
    ("chat_template_kwargs enable_thinking=true", {"chat_template_kwargs": {"enable_thinking": True}}, True),
    ("chat_template_kwargs enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, False),
    ("reasoning_effort=high", {"reasoning_effort": "high"}, True),
    ("reasoning_effort=low", {"reasoning_effort": "low"}, True),
    ("reasoning_effort=none", {"reasoning_effort": "none"}, False),
]
print("== non-streaming")
default_thinks = None
for label, extra, want in cases:
    r, c, _ = msg(post({**base, **extra}))
    if want is None:
        default_thinks = bool(r.strip()); print(f"INFO {label:48s} -> thinking {'ON' if default_thinks else 'OFF'} by default")
        want = default_thinks
    check(label, r, c, want, "391")
print("== streaming")
for label, extra, want in cases[1:3]:
    r, c, _ = stream({**base, **extra})
    check(label + " (stream)", r, c, want, "391")
print("== tool call with thinking on")
tb = {"model": A.model, "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}], "tools": TOOLS,
      "max_tokens": 2048, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}
r, c, t = msg(post(tb)); check("tool call, thinking on", r, c, True, tools=t, want_tool="get_weather")
r, c, t = stream(tb); check("tool call, thinking on (stream)", r, c, True, tools=t, want_tool="get_weather")
tb["chat_template_kwargs"] = {"enable_thinking": False}
r, c, t = stream(tb); check("tool call, thinking off (stream)", r, c, False, tools=t, want_tool="get_weather")
print("== multi-turn, thinking on, previous reasoning passed back")
r1, c1, _ = msg(post({**base, "chat_template_kwargs": {"enable_thinking": True}}))
mt = {"model": A.model, "max_tokens": 2048, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True},
      "messages": [{"role": "user", "content": Q}, {"role": "assistant", "content": c1, "reasoning_content": r1},
                   {"role": "user", "content": "Now add 9 to that. Number only."}]}
r, c, _ = msg(post(mt)); check("second turn (reasoning_content in history)", r, c, True, "400")
mt["messages"][1] = {"role": "assistant", "content": c1}
r, c, _ = msg(post(mt)); check("second turn (no reasoning in history)", r, c, True, "400")
print(f"\n{'ALL OK' if fails == 0 else str(fails) + ' FAILED'}")

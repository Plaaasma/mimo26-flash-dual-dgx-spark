#!/usr/bin/env python3
"""structured_check.py: strict JSON-schema responses with thinking on/off, plain and right after a tool round-trip
(the turn where MiMo can emit the JSON without closing its reasoning, so vLLM's parser files it under
reasoning_content and content comes back empty). N runs per case at temperature 1 (the server default sampling)."""
import argparse, json, urllib.request
P = argparse.ArgumentParser(); P.add_argument("--url", default="http://127.0.0.1:8888"); P.add_argument("--model", default="mimo-v2.6-flash")
P.add_argument("-n", type=int, default=5); A = P.parse_args()
SCHEMA = {"type": "json_schema", "json_schema": {"name": "city_report", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["city", "temperature_c", "summary"],
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "number"}, "summary": {"type": "string"}}}}}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
PLAIN = [{"role": "user", "content": "Give me a short weather report for Paris (assume 18 C and light rain)."}]
POST_TOOL = [
    {"role": "user", "content": "What's the weather in Paris? Use the tool, then answer as the requested JSON."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"city\": \"Paris\", \"temp_c\": 18, \"conditions\": \"light rain\"}"},
]
def call(messages, thinking, tools):
    body = {"model": A.model, "messages": messages, "max_tokens": 1024, "response_format": SCHEMA,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools: body["tools"] = TOOLS
    r = json.load(urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                                headers={"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]; content = (m.get("content") or "").strip(); reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    try:
        obj = json.loads(content); ok = set(obj) == {"city", "temperature_c", "summary"}
    except Exception:
        ok = False
    return ok, content, reasoning
for label, msgs, tools in (("plain", PLAIN, False), ("post-tool", POST_TOOL, True)):
    for thinking in (False, True):
        res = [call(msgs, thinking, tools) for _ in range(A.n)]
        good = sum(r[0] for r in res); empty = sum(1 for r in res if not r[1])
        json_in_reasoning = sum(1 for r in res if not r[1] and "{" in r[2])
        print(f"{label:9s} thinking={'on ' if thinking else 'off'}: valid JSON in content {good}/{A.n} | empty content {empty} "
              f"(JSON found in reasoning {json_in_reasoning})")
        bad = [r for r in res if not r[0]]
        if bad: print(f"   e.g. content={bad[0][1][:100]!r} reasoning={bad[0][2][-160:]!r}")

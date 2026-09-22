#!/usr/bin/env bash
# smoke.sh [base_url] [model]: chat, reasoning split, tool call, and a synthetic image through the OpenAI API.
set -u
URL="${1:-http://127.0.0.1:8888}"; MODEL="${2:-mimo-v2.6-flash}"; KEY="${VLLM_API_KEY:-}"
H=(-H "Content-Type: application/json"); [ -n "$KEY" ] && H+=(-H "Authorization: Bearer $KEY")
j() { curl -sS -m 600 "${H[@]}" "$URL/v1/chat/completions" -d "$1"; }
echo "== models"; curl -sS -m 10 "${H[@]}" "$URL/v1/models" | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
echo "== chat (thinking off)"
j "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":32,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m.get("content")), "| reasoning:", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:60]), "| usage:", r["usage"])'
echo "== chat (thinking on)"
j "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":512,\"chat_template_kwargs\":{\"enable_thinking\":true}}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m.get("content")), "| reasoning chars:", len(m.get("reasoning_content") or m.get("reasoning") or ""), "| reasoning head:", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:120]), "| usage:", r["usage"])'
echo "== tool call"
j "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"max_tokens\":256,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("tool_calls:", [(t["function"]["name"], t["function"]["arguments"]) for t in (m.get("tool_calls") or [])], "| content:", repr((m.get("content") or "")[:80]))'
echo "== vision (synthetic PNG: red circle, blue square, text CODE 7482)"
IMG=$(python3 - <<'PY' 2>/dev/null
import base64, io
try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit
im = Image.new("RGB", (256, 160), "white"); d = ImageDraw.Draw(im)
d.ellipse((20, 30, 80, 90), fill="red"); d.rectangle((110, 30, 170, 90), fill="blue"); d.text((20, 120), "CODE 7482", fill="black")
b = io.BytesIO(); im.save(b, "PNG"); print(base64.b64encode(b.getvalue()).decode())
PY
)
if [ -n "$IMG" ]; then
  j "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG\"}},{\"type\":\"text\",\"text\":\"Describe the shapes, their colours and any text.\"}]}],\"max_tokens\":200,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print("content:", repr((r["choices"][0]["message"].get("content") or "")[:300]), "| usage:", r["usage"])'
else
  echo "(PIL missing on this host — vision test skipped)"
fi
echo "== long prompt needle (~4.5K tokens: exercises the prefill attention paths)"
python3 - "$URL" "$MODEL" "$KEY" <<'PY'
import json, sys, urllib.request, random
url, model, key = sys.argv[1], sys.argv[2], sys.argv[3]
random.seed(1); words = ["alpha","bridge","carbon","delta","ember","falcon","granite","harbor"]
doc = " ".join(random.choice(words) + str(random.randrange(100)) for _ in range(1500))
body = {"model": model, "messages": [{"role": "user", "content": "Here is a list of codes:\n" + doc + "\n\nWhat was the very first code in the list? Reply with just the code."}],
        "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
h = {"Content-Type": "application/json"}
if key: h["Authorization"] = "Bearer " + key
r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h), timeout=900))
got = r["choices"][0]["message"]["content"].strip(); exp = doc.split()[0]
print(f"expected {exp} | model {got!r} | prompt_tokens {r['usage']['prompt_tokens']} |", "OK" if exp in got else "WRONG")
PY

#!/usr/bin/env python3
"""agent_check.py: does the model stay on task in a coding-agent conversation?

Replays what an agent harness sends after "continue my project" and three file reads (tool calls + tool results),
with harness-style sampling (temperature 1.0, top_p 0.95), N times. A sample passes when its thinking or first action
is about the project's TODO items. With the UNCENSORED checkpoint's own chat template (a thinking prefill that calls
every request "a research or creative-writing question") 0 of 4 samples passed: the thinking went to Aristotle's
Poetics, Shapley values and the history of incorporation. With MiMo's original template 4 of 4 passed.

  tests/agent_check.py --url http://HEAD:8888 [--model mimo-v2.6-flash] [-n 6]
"""
import argparse, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

A = argparse.ArgumentParser()
A.add_argument("--url", default="http://127.0.0.1:8888"); A.add_argument("--model", default="mimo-v2.6-flash")
A.add_argument("-n", type=int, default=6)
A = A.parse_args()

FILES = {
    "TODO.md": "# TODO\n\n- [x] add / take / list commands\n- [x] persist to inventory.json\n"
               "- [ ] `low` command: list items whose quantity is below their `min`, sorted by how far below they are\n"
               "- [ ] `export` command: write the inventory to CSV (`name,quantity,min`) at a given path\n"
               "- [ ] `take` must refuse to go below zero\n",
    "stockkeeper/store.py": "import json\nfrom dataclasses import dataclass, asdict\nfrom pathlib import Path\n\n\n@dataclass\n"
                            "class Item:\n    name: str\n    quantity: int\n    min: int = 0\n\n\nclass Store:\n"
                            "    def __init__(self, path):\n        self.path = Path(path)\n        self.items = {}\n"
                            "        if self.path.exists():\n            for raw in json.loads(self.path.read_text()):\n"
                            "                self.items[raw['name']] = Item(**raw)\n\n    def save(self):\n"
                            "        self.path.write_text(json.dumps([asdict(i) for i in self.items.values()], indent=2))\n\n"
                            "    def take(self, name, quantity):\n        item = self.items[name]\n        item.quantity -= quantity\n"
                            "        return item\n",
    "stockkeeper/__main__.py": "import argparse\nfrom pathlib import Path\nfrom .store import Store\n\n\ndef main(argv=None):\n"
                               "    p = argparse.ArgumentParser(prog='stockkeeper')\n    sub = p.add_subparsers(dest='cmd', required=True)\n"
                               "    t = sub.add_parser('take'); t.add_argument('name'); t.add_argument('quantity', type=int)\n"
                               "    sub.add_parser('list')\n    args = p.parse_args(argv)\n    store = Store(Path('inventory.json'))\n"
                               "    if args.cmd == 'take':\n        print(store.take(args.name, args.quantity))\n    store.save()\n",
}
fn = lambda name, desc, props: {"type": "function", "function": {"name": name, "description": desc, "parameters": {
    "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": props}}}
TOOLS = [fn("read", "Read a file", ["filePath"]), fn("edit", "Replace text in a file", ["filePath", "oldString", "newString"]),
         fn("bash", "Run a shell command", ["command"])]
msgs = [{"role": "system", "content": "You are a CLI coding agent working in the user's repository. Use the tools to read and edit "
                                      "files and run commands."},
        {"role": "user", "content": "continue my project"}]
for i, (path, text) in enumerate(FILES.items()):
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                 "name": "read", "arguments": json.dumps({"filePath": path})}}]},
             {"role": "tool", "tool_call_id": f"c{i}", "content": text}]
KEYS = ("low", "below", "min", "export", "csv", "negative", "zero", "todo", "take", "store", "test")


def one(_):
    body = {"model": A.model, "messages": msgs, "tools": TOOLS, "max_tokens": 1200, "temperature": 1.0, "top_p": 0.95}
    r = json.load(urllib.request.urlopen(urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                                headers={"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]
    text = ((m.get("reasoning_content") or m.get("reasoning") or "") + " " + (m.get("content") or "") + " "
            + " ".join(t["function"]["name"] + " " + t["function"]["arguments"] for t in (m.get("tool_calls") or []))).lower()
    return any(k in text for k in KEYS), re.sub(r"\s+", " ", text)[:120]


with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(one, range(A.n)))
for ok, head in res:
    print(f"{'OK  ' if ok else 'FAIL'} {head}")
passed = sum(ok for ok, _ in res)
print(f"\n{passed}/{A.n} on task" + ("" if passed == A.n else "  <-- the model drifts off task (check the chat template / sampling defaults)"))

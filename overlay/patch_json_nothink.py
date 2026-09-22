#!/usr/bin/env python3
"""Strict JSON responses from MiMo: render requests with a JSON response_format on the no-thinking template path.

With thinking on, vLLM's `mimo` reasoning parser (the generic Qwen3 adapter) files every token as reasoning until
it sees the end-of-thinking marker, and structured-output grammar enforcement waits for that same marker. On some
turns (typically right after a tool result) MiMo emits the requested JSON without closing its reasoning: the grammar
never engages and the answer lands in `reasoning_content` with `content` empty. When a request carries a
`response_format` of type json_schema / json_object, this sets `enable_thinking: false` in its chat template kwargs
(template + parser agree; ordinary requests are untouched). Gate: MIMO26_JSON_NOTHINK=1.
Idempotent; fails closed on anchor drift.
"""
import os
import sys
from pathlib import Path

MARK = "# [mimo26-json-nothink]"
P = Path(os.environ.get("MIMO26_CHAT_PROTOCOL_PY",
                        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py"))
OLD = """        return ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                extra_kwargs,
            ),
"""
NEW = """        import os as _m26_os  """ + MARK + """
        _m26_rf = getattr(self.response_format, "type", None)
        if _m26_os.environ.get("MIMO26_JSON_NOTHINK", "0") == "1" and _m26_rf in ("json_schema", "json_object"):
            extra_kwargs["enable_thinking"] = False
            self.chat_template_kwargs = {**(self.chat_template_kwargs or {}), "enable_thinking": False}
        return ChatParams(
            chat_template=self.chat_template or default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                self.chat_template_kwargs,
                extra_kwargs,
            ),
"""


def main() -> int:
    t = P.read_text()
    if MARK in t:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    if t.count(OLD) != 1:
        raise SystemExit(f"{P}: expected exactly one anchor, found {t.count(OLD)} — refusing to patch")
    P.write_text(t.replace(OLD, NEW, 1))
    print(f"patched {P.name} (JSON response_format -> no-thinking template path when MIMO26_JSON_NOTHINK=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

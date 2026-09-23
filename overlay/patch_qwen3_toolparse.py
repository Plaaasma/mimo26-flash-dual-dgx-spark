#!/usr/bin/env python3
"""Tool-call arguments that contain the tool-call markup no longer get truncated (qwen3 / `mimo` tool parser).

Symptom: a string argument whose value contains a literal `</parameter>` (e.g. code with that tag in a string or
comment) came back cut at that tag; a value containing `</function>` (e.g. `y = "</function></tool_call>"`) ended
the whole call there, and in streaming the rest of the value leaked into `content`. A `write`/`edit` of such a file
silently wrote a truncated file. A literal `<parameter=x>` inside a value split it into a bogus second argument.

Cause (vllm/parser/qwen3.py + vllm/parser/engine/streaming_parser_engine.py): `_PARAM_RE` ends a value at the first
`</parameter>` (non-greedy) or at the next `<parameter=`, and the state machine turns the first `</function>` seen in
TOOL_ARGS into TOOL_CALL_END, wherever they occur.

Fix (structure-aware, byte-identical for well-formed calls without such literals):
  * a `</parameter>` ends a value only when followed by optional whitespace and then `<parameter=`, `</function>` or
    the end of the arguments; any other `</parameter>` is part of the value;
  * a `<parameter=` ends an unclosed value only at the start of a line with no `</parameter>` before it (the upstream
    tolerance for a model that drops `</parameter>` between arguments); mid-line it is part of the value;
  * in TOOL_ARGS, `</function>` closes the call only right after `<function=name>` (no-arg call) or right after a
    `</parameter>` (plus whitespace); otherwise it is argument text. A `</tool_call>` that follows such a literal
    `</function>` on its own line closes the call (a model that dropped the last `</parameter>`), and a trailing
    `</function>` line is not part of that last value.
Streaming stays prefix-stable: the partial and final conversions apply the same rules, so streamed argument deltas
are always a prefix of the final JSON.
Known limit: a value containing `</parameter>` immediately followed (after whitespace) by `<parameter=` or
`</function>` (e.g. a document that quotes a complete tool call) is still indistinguishable from the real end.
Gate: MIMO26_TOOLPARSE_FIX (default 1). Idempotent; fails closed on anchor drift.
"""
import os
import sys
from pathlib import Path

MARK = "# [mimo26-toolparse]"
V = Path(os.environ.get("MIMO26_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"))
QWEN3 = V / "parser/qwen3.py"
ENGINE = V / "parser/engine/streaming_parser_engine.py"

# ---------------------------------------------------------------- qwen3.py: argument converter
CONV_OLD = '''def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}

    for match in _PARAM_RE.finditer(raw_args):
        name = match.group(1)
        value = match.group(2)
        params[name] = _trim_wrapping_newlines(value)

    if partial:
        remaining = _PARAM_RE.sub("", raw_args)
        m = _PARTIAL_PARAM_RE.search(remaining)
        if m:
            name = m.group(1)
            value = m.group(2)
            if name:
                params[name] = _trim_wrapping_newlines(value)

    return json.dumps(params, ensure_ascii=False)
'''
CONV_NEW = '''_M26_OPEN_RE = re.compile(r"<\\s*parameter\\s*=\\s*([^>]*)>")  ''' + MARK + '''
_M26_CLOSE_RE = re.compile(r"<\\s*/\\s*parameter\\s*>")
_M26_OPEN_START_RE = re.compile(r"<\\s*parameter\\s*=")  # an opening tag as soon as it starts (name may still stream)
# what may follow a </parameter> that really ends a value
_M26_REAL_CLOSE_TAIL_RE = re.compile(r"\\s*(?:<\\s*parameter\\s*=|<\\s*/\\s*function\\s*>|\\Z)")
# a trailing </function> line of an unclosed last value (the call was closed by </tool_call> or the end of output)
_M26_TRAIL_FUNC_RE = re.compile(r"\\n?[ \\t]*<\\s*/\\s*function\\s*>\\s*\\Z")


def _qwen3_arg_converter(raw_args: str, partial: bool) -> str:
    """Parse ``<parameter=NAME>VALUE</parameter>`` into a JSON object.

    A ``</parameter>`` ends a value only when followed by whitespace and then
    ``<parameter=``, ``</function>`` or the end of the arguments; any other
    ``</parameter>`` (and a mid-line ``<parameter=``) is part of the value, so
    code that contains the markup survives. A line-start ``<parameter=`` before
    any ``</parameter>`` still starts the next argument (a model that dropped the
    closing tag). Partial and final parses use the same rules, which keeps the
    streamed JSON a prefix of the final one.
    """
    params: dict[str, object] = {}
    pos = 0
    while True:
        m = _M26_OPEN_RE.search(raw_args, pos)
        if m is None:
            break
        name, vstart = m.group(1), m.end()
        first_close = _M26_CLOSE_RE.search(raw_args, vstart)
        real_close = None
        c = first_close
        while c is not None:
            if _M26_REAL_CLOSE_TAIL_RE.match(raw_args, c.end()):
                real_close = c
                break
            c = _M26_CLOSE_RE.search(raw_args, c.end())
        nxt = _M26_OPEN_START_RE.search(raw_args, vstart)
        if (
            nxt is not None
            and (first_close is None or nxt.start() < first_close.start())
            and raw_args[vstart : nxt.start()].rstrip(" \\t").endswith("\\n")
        ):
            # missing </parameter>: a line-start <parameter= opens the next argument
            params[name] = _trim_wrapping_newlines(raw_args[vstart : nxt.start()])
            pos = nxt.start()
            continue
        if real_close is not None:
            params[name] = _trim_wrapping_newlines(raw_args[vstart : real_close.start()])
            pos = real_close.end()
            continue
        # unclosed last value: in progress (partial) or a call that ended without </parameter>
        value = _M26_TRAIL_FUNC_RE.sub("", raw_args[vstart:])
        if name:
            params[name] = _trim_wrapping_newlines(value)
        break

    return json.dumps(params, ensure_ascii=False)
'''

# ---------------------------------------------------------------- streaming_parser_engine.py: </function> guard
RESET_OLD = "        self._args_escape_next: bool = False\n"
RESET_NEW = RESET_OLD + "        self._m26_args_text: str = \"\"  " + MARK + " raw text of the current call's arguments\n"

TERM_OLD = '''    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        key = (self.state, terminal)
'''
TERM_NEW = '''    def _on_terminal(
        self, terminal: str, value: str, token_count: int = 0
    ) -> list[SemanticEvent]:
        if (
            self.state == ParserState.TOOL_ARGS
            and terminal in ("FUNC_END", "TOOL_END")
            and _m26_toolparse_active(self)
        ):  ''' + MARK + '''
            args_text = self._m26_args_text
            if terminal == "FUNC_END" and not _m26_func_end_closes(args_text):
                # </function> inside an argument value: keep it as argument text
                return self._emit_for_state(value, token_count)
            if terminal == "TOOL_END" and _M26_FUNC_LINE_END_RE.search(args_text):
                # </function> line + </tool_call>: the model dropped the last </parameter>; close the call
                return self._apply_transition(
                    Transition(ParserState.CONTENT, (EventType.TOOL_CALL_END,)),
                    value,
                    token_count,
                )
        key = (self.state, terminal)
'''

EMIT_OLD = '''        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)
'''
EMIT_NEW = '''        if self.state == ParserState.TOOL_ARGS:
            self._m26_args_text += text  ''' + MARK + '''
            if self.config.tool_args_json:
                return self._feed_args_text(text)
'''

APPLY_OLD = '''        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
'''
APPLY_NEW = '''        if self.state == ParserState.TOOL_ARGS:  ''' + MARK + '''
            if previous_state == ParserState.TOOL_ARGS:
                self._m26_args_text += value  # <parameter= / </parameter> markup
            else:
                self._m26_args_text = ""
        if self.state == ParserState.TOOL_ARGS:
            self._args_brace_depth = 0
'''

ENGINE_HELPERS = '''


''' + MARK + ''' helpers
import re as _m26_re

_M26_PARAM_CLOSE_END_RE = _m26_re.compile(r"<\\s*/\\s*parameter\\s*>\\s*\\Z")
_M26_FUNC_LINE_END_RE = _m26_re.compile(r"(?:\\A|\\n)[ \\t]*<\\s*/\\s*function\\s*>\\s*\\Z")


def _m26_toolparse_active(engine) -> bool:
    """Only the qwen3 XML grammar (raw-text arguments + _qwen3_arg_converter), gate MIMO26_TOOLPARSE_FIX."""
    active = getattr(engine, "_m26_tp_active", None)
    if active is None:
        import os as _os

        cfg = engine.config
        active = (
            _os.environ.get("MIMO26_TOOLPARSE_FIX", "1") == "1"
            and getattr(cfg.arg_converter, "__name__", "") == "_qwen3_arg_converter"
            and not cfg.tool_args_json
        )
        engine._m26_tp_active = active
    return active


def _m26_func_end_closes(args_text: str) -> bool:
    """</function> closes the call right after <function=name> or right after a </parameter> (plus whitespace)."""
    return not args_text.strip() or _M26_PARAM_CLOSE_END_RE.search(args_text) is not None
'''


def patch_file(path: Path, edits, append: str = "") -> None:
    text = path.read_text()
    if MARK in text:
        print(f"{path.name}: {MARK} already present — skipping")
        return
    for label, old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one anchor for {label}, found {n} — refusing to patch")
        text = text.replace(old, new, 1)
    path.write_text(text + append)
    print(f"patched {path.name}: " + ", ".join(label for label, _, _ in edits))


def main() -> int:
    if os.environ.get("MIMO26_TOOLPARSE_FIX", "1") != "1":
        print("patch_qwen3_toolparse: MIMO26_TOOLPARSE_FIX != 1 — tool parser left stock")
        return 0
    # check every anchor before writing anything, so a drifted engine file cannot leave qwen3.py half-patched
    for path, edits in ((QWEN3, [("arg-converter", CONV_OLD, None)]),
                        (ENGINE, [("args-reset", RESET_OLD, None), ("func-end-guard", TERM_OLD, None),
                                  ("args-text", EMIT_OLD, None), ("args-markup", APPLY_OLD, None)])):
        text = path.read_text()
        if MARK in text:
            continue
        for label, old, _ in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{path}: expected exactly one anchor for {label}, found {text.count(old)} — refusing to patch")
    patch_file(QWEN3, [("arg-converter", CONV_OLD, CONV_NEW)])
    patch_file(ENGINE, [("args-reset", RESET_OLD, RESET_NEW), ("func-end-guard", TERM_OLD, TERM_NEW),
                        ("args-text", EMIT_OLD, EMIT_NEW), ("args-markup", APPLY_OLD, APPLY_NEW)], ENGINE_HELPERS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

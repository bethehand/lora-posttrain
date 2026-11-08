"""OLMo 3 tool-call parser for prime-rl's renderers package, registered as tool_parser "olmo3".

OLMo 3 writes tool calls as Python calls inside two non-special tokens:
``<function_calls>name(key="value")\nname2(...)</function_calls>``. Training has to read a
completion the way evaluation does, where vLLM's ``olmo3`` tool parser (Olmo3PythonicToolParser,
vLLM 0.26) turns the response text into tool calls. This is a port of its non-streaming path:

- the text is the completion decoded without special tokens, as vLLM's chat endpoint passes it;
- only the first closed <function_calls>...</function_calls> block is read; without one, the
  whole text is tried;
- the lines are joined with ", ", wrapped in [...], checked against vLLM's regex (1 s timeout)
  and parsed with ast: every element must be a call with a plain or dotted name, and keyword
  values must be literals (null, true and false are accepted);
- on success the text outside the calls is dropped, as vLLM returns content=None; on failure
  there are no tool calls and the reply is the full text, so the episode ends as it would at eval.

A failed attempt that contains <function_calls> is recorded with name=None, which verifiers
drops, and no token span, so it is visible on the parsed response but never executed or masked.

Installed into prime-rl by setup/install_olmo3_parser.sh; scripts/olmo3_parity.py checks it
against vLLM on real OLMo completions.
"""

from __future__ import annotations

import ast
import json

import regex

from renderers.base import ParsedToolCall, ToolCallParseStatus

# Olmo3PythonicToolParser.TOOL_CALL_REGEX and VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS in vLLM 0.26
TOOL_CALL_REGEX = regex.compile(
    r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
    regex.DOTALL,
)
REGEX_TIMEOUT_S = 1
BLOCK = regex.compile(r"<function_calls>(.*?)</function_calls>", regex.DOTALL)
JSON_NAME_LITERALS = {"null": None, "true": True, "false": False}


class Unparsed(Exception):
    pass


def parameter_value(node: ast.expr):
    """vllm.tool_parsers.utils.get_parameter_value."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in node.keys):
            raise Unparsed("dict keys must be literals")
        return {k.value: parameter_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [parameter_value(v) for v in node.elts]
    if isinstance(node, ast.Name) and node.id in JSON_NAME_LITERALS:
        return JSON_NAME_LITERALS[node.id]
    raise Unparsed("arguments must be literals")


def call_name(node: ast.expr) -> str:
    """vllm.tool_parsers.utils._ast_callable_dotted_name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise Unparsed("call target must be a name")
    parts.append(node.id)
    return ".".join(reversed(parts))


def parse_text(text: str) -> list[tuple[str, dict]] | None:
    """The (name, arguments) of each call vLLM's olmo3 parser finds in `text`, or None when it
    finds no tool call."""
    match = BLOCK.search(text)
    body = match.group(1).strip() if match else text
    body = "[" + ", ".join(line.strip() for line in body.splitlines() if line.strip()) + "]"
    try:
        if TOOL_CALL_REGEX.match(body, timeout=REGEX_TIMEOUT_S) is None:
            return None
    except TimeoutError:
        return None
    try:
        parsed = getattr(ast.parse(body).body[0], "value", None)
        if not isinstance(parsed, ast.List) or not all(isinstance(e, ast.Call) for e in parsed.elts):
            raise Unparsed("not a list of calls")
        calls = []
        for call in parsed.elts:
            name = call_name(call.func)
            arguments = {kw.arg: parameter_value(kw.value) for kw in call.keywords}
            json.dumps(arguments, ensure_ascii=False)  # vLLM serializes here; a failure means no call
            calls.append((name, arguments))
        return calls
    except Exception:
        return None


class Olmo3ToolParser:
    """``<function_calls>`` pythonic tool calls, read as vLLM's ``olmo3`` tool parser reads them."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        text = self._tokenizer.decode(ids, skip_special_tokens=True)
        calls = parse_text(text)
        if calls is not None:
            return [], [ParsedToolCall(raw=text, name=name, arguments=arguments) for name, arguments in calls]
        if "<function_calls>" not in text:
            return ids, []
        status = (ToolCallParseStatus.MALFORMED_STRUCTURE if "</function_calls>" in text
                  else ToolCallParseStatus.UNCLOSED_BLOCK)
        return ids, [ParsedToolCall(raw=text, status=status)]


class Olmo3KeepTextToolParser(Olmo3ToolParser):
    """The same, keeping the text before a successful call as setup/vllm_olmo3_keep_text.py does at eval."""

    def extract(self, ids: list[int]) -> tuple[list[int], list[ParsedToolCall]]:
        content_ids, parsed = super().extract(ids)
        if content_ids or not parsed:
            return content_ids, parsed
        text = self._tokenizer.decode(ids, skip_special_tokens=True)
        head = text.split("<function_calls>", 1)[0] if "<function_calls>" in text else ""
        if not head.strip():
            return [], parsed
        return self._tokenizer.encode(head, add_special_tokens=False), parsed

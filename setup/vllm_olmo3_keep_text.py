"""vLLM's OLMo 3 tool parser, keeping the text the model writes before a tool call.

vLLM's own parser returns content=None whenever it finds a call, so that text is gone from the
next turn's prompt. Chapter 3's model reasons in that text and has to read it back later. Chapter
2's results used the original `olmo3` parser, which stays as it is. Serve with:

    --vllm.tool-call-parser olmo3_keep_text --vllm.tool-parser-plugin setup/vllm_olmo3_keep_text.py
"""
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.olmo3_tool_parser import Olmo3PythonicToolParser


@ToolParserManager.register_module("olmo3_keep_text")
class Olmo3KeepTextToolParser(Olmo3PythonicToolParser):
    def extract_tool_calls(self, model_output, request):
        info = super().extract_tool_calls(model_output, request)
        if info.tools_called and "<function_calls>" in model_output:
            head = model_output.split("<function_calls>", 1)[0]
            if head.strip():
                info.content = head
        return info

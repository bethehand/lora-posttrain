#!/usr/bin/env python3
"""A model folder for OLMo 3 Think that can use tools: its weights, with Instruct's chat template.

Think's own template ignores the API's tools argument and drops assistant tool calls and tool
results, so through the chat API it can never see a tool. Instruct's template renders all of these
in the format both models were built around. The only change to it is the generation prompt,
which opens with <think> as Think's own template does.

    python setup/make_olmo3_think_tools.py <think dir> <instruct dir> <out dir>
"""
import json
import os
import sys

think, instruct, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for name in os.listdir(think):
    if name in ("chat_template.jinja", "tokenizer_config.json") or name.startswith("."):
        continue
    target = os.path.join(out, name)
    if not os.path.lexists(target):
        os.symlink(os.path.abspath(os.path.join(think, name)), target)

template = open(os.path.join(instruct, "chat_template.jinja")).read()
opening = "{{- '<|im_start|>assistant\n' -}}{%- endif -%}{%- endfor -%}"
if template.count(opening) != 1:
    raise SystemExit("Instruct's generation prompt is not where this script expects it")
template = template.replace(opening, "{{- '<|im_start|>assistant\n<think>' -}}{%- endif -%}{%- endfor -%}")
with open(os.path.join(out, "chat_template.jinja"), "w") as f:
    f.write(template)

with open(os.path.join(think, "tokenizer_config.json")) as f:
    config = json.load(f)
config.pop("chat_template", None)
with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("wrote", out)

"""One row through the patched dataset code, with and without a turn left out of training."""
import json, os, sys
sys.dont_write_bytecode = True
# A training run may be sharing the machine: tokenizers would otherwise take every core, and training slowed to half its speed.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.nice(19)
from transformers import AutoTokenizer
from renderers.configs import DefaultRendererConfig
from renderers.default import DefaultRenderer
from prime_rl.configs.sft import LossMaskConfig
from prime_rl.trainer.sft.data import SFTDataset
tok = AutoTokenizer.from_pretrained("/mnt/nvme1/bruce/models/Olmo-3-7B-Instruct")
d = SFTDataset.__new__(SFTDataset)
d.renderer = DefaultRenderer(tok, DefaultRendererConfig(tool_parser="olmo3_keep_text"))
d.loss_mask_config = LossMaskConfig()
d.seq_len = 8192
d.multimodal = False
import logging
d.logger = logging.getLogger("x")
# The trainer reads the folder with this loader, which fixes its schema from the first rows.
from datasets import load_dataset
print("the trainer's loader reads", len(load_dataset(os.path.dirname(os.path.abspath(sys.argv[1])), split="train")), "rows")
rows = [json.loads(l) for l in open(sys.argv[1])]
def spans(sample):
    ids, mask, out, i = sample["target_ids"], sample["loss_mask"], [], 0
    while i < len(ids):
        if mask[i]:
            j = i
            while j < len(ids) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return ids, out
for row in rows[:1]:
    s = d._process(row)
    ids, sp = spans(s)
    print("tokens", len(ids), "trained", sum(s["loss_mask"]), "spans", len(sp), "weights", set(s["loss_weights"]))
    for a, b in sp[:2]:
        print("  starts", [tok.decode([t]) for t in ids[a:a + 4]], "ends", [tok.decode([t]) for t in ids[b - 3:b]])
    print("  before the first span:", [tok.decode([t]) for t in ids[sp[0][0] - 4:sp[0][0]]])
broken = json.loads(json.dumps(rows[0]))
first = next(i for i, m in enumerate(broken["messages"]) if m["role"] == "assistant")
broken["messages"][first]["train"] = False
broken["weight"] = -0.25
s2 = d._process(broken)
ids2, sp2 = spans(s2)
print("marked: same tokens", ids2 == ids, "trained", sum(s2["loss_mask"]), "spans", len(sp2), "weights", set(s2["loss_weights"]))
print("all rows: trained share", round(sum(sum(d._process(r)["loss_mask"]) for r in rows) / sum(len(d._process(r)["loss_mask"]) for r in rows), 3), "longest row", max(len(d._process(r)["input_ids"]) for r in rows))
# what a live server would be asked to continue, against the row's context
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
msgs = deserialize_tool_calls(normalize_messages(rows[0]["messages"], default_role="assistant"))
turn = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
prompt = d.renderer.render(msgs[:turn], tools=json.loads(rows[0]["tools"]), add_generation_prompt=True).token_ids
full = s["input_ids"]
print("live prompt for the first turn is a prefix of the row:", list(prompt) == full[:len(prompt)], "; first trained target is right after it:", sp[0][0] == len(prompt) - 1)

#!/usr/bin/env python3
"""Four small additions to prime-rl's trainers. Without the new inputs it behaves as before.

- A dataset row may carry a `weight`, multiplying the loss on its tokens. Offline RL is SFT with
  each trajectory weighted by its advantage, negative weights included.
- The default renderer, which OLMo uses, has no record of which tokens a model generates, and
  prime-rl refuses to train with it unless told by role. Trained tokens are now the assistant
  turns without the `<|im_start|>assistant` line the template writes, which is what a live model
  generates. An assistant message with `"train": false` stays in the context but out of the loss:
  a trace that starts from a student's broken formula must not teach the student to write it.
- LORA_POSTTRAIN_ADAPTER_EVERY=N writes the adapter every N steps and at the end, as RL runs do.
  prime-rl only writes one when online evals are on.
- LORA_POSTTRAIN_INIT_ADAPTER=<dir> starts SFT or RL from that adapter instead of a fresh one, with
  a new optimizer and step count. Each stage of the recipe starts from the file the stage before
  it was evaluated on. prime-rl's own resume can't do this: an RL resume needs orchestrator state
  an SFT checkpoint doesn't have, and it keeps counting steps.

Idempotent: applies to prime-rl v0.9.0 (ab5de8f) and stops if the code isn't what it expects.
    python setup/patch_prime_rl_sft.py <prime-rl dir>
"""
import pathlib
import sys

MARKER = "# lora-posttrain: per-token loss weights"
EDITS = {
    "src/prime_rl/trainer/lora.py": [
        ('''        for _, module in self._modules:
            module.reset_parameters(0)
''',
         f'''        for _, module in self._modules:
            module.reset_parameters(0)
        {MARKER}: a stage of the recipe continues from the adapter the stage before it ended with.
        start = __import__("os").environ.get("LORA_POSTTRAIN_INIT_ADAPTER")
        if start:
            self._load_adapter(start)

    def _load_adapter(self, path: str) -> None:
        from safetensors.torch import load_file
        from torch.distributed.tensor import DTensor, distribute_tensor

        saved = load_file(f"{{path}}/adapter_model.safetensors")
        for prefix, module in self._modules:
            for name, param in module.named_parameters_for_adapter(0):
                value, target = saved.pop(f"{{prefix}}.{{name}}.weight"), param.detach()
                if value.shape != target.shape:
                    raise ValueError(f"{{prefix}}.{{name}}: adapter has {{value.shape}}, the model wants {{target.shape}}")
                value = value.to(target.device, target.dtype)
                if isinstance(target, DTensor):
                    target.to_local().copy_(distribute_tensor(value, target.device_mesh, target.placements).to_local())
                else:
                    target.copy_(value)
        if saved:
            raise ValueError(f"{{len(saved)}} tensors in {{path}} have no place in the model, such as {{next(iter(saved))}}")
        get_logger().info(f"Started the adapter from {{path}}")
'''),
    ],
    "src/prime_rl/transports/weights/base.py": [
        ('''        ready_file = step_dir / RECEIVER_READY_MARKER
''',
         f'''        {MARKER}: with LORA_POSTTRAIN_ADAPTER_EVERY nobody is listening, the files are the point.
        if __import__("os").environ.get("LORA_POSTTRAIN_ADAPTER_EVERY"):
            return
        ready_file = step_dir / RECEIVER_READY_MARKER
'''),
        ('''        broadcast_dir = get_broadcast_dir(self.output_dir)
        for old_step in get_all_ckpt_steps(broadcast_dir):
''',
         '''        if __import__("os").environ.get("LORA_POSTTRAIN_ADAPTER_EVERY"):
            return
        broadcast_dir = get_broadcast_dir(self.output_dir)
        for old_step in get_all_ckpt_steps(broadcast_dir):
'''),
    ],
    "packages/prime-rl-configs/src/prime_rl/configs/sft.py": [
        ("""        elif isinstance(self.renderer, DefaultRendererConfig):
            reason = "renderer.name='default' selects DefaultRenderer"
""",
         f"""        elif isinstance(self.renderer, DefaultRendererConfig):
            {MARKER}: OLMo's template has no typed renderer, and data.py now marks what a model
            # generates. Its tokens and mask are checked against live prompts by check_sft_dataset.py.
            if self.renderer.tool_parser in ("olmo3", "olmo3_keep_text"):
                return self
            reason = "renderer.name='default' selects DefaultRenderer"
"""),
    ],
    "src/prime_rl/trainer/sft/data.py": [
        ("class Sample(TypedDict):\n    input_ids: list[int]\n    position_ids: list[int]\n    loss_mask: list[bool]\n",
         f"{MARKER}\nclass Sample(TypedDict):\n    input_ids: list[int]\n    position_ids: list[int]\n    loss_mask: list[bool]\n    loss_weights: list[float]\n"),
        ('    loss_mask: Bool[Tensor, "batch seq"]\n    seq_lens: Int[Tensor, "packed"]\n',
         '    loss_mask: Bool[Tensor, "batch seq"]\n    loss_weights: Tensor\n    seq_lens: Int[Tensor, "packed"]\n'),
        ('                "position_ids": position_ids,\n                "loss_mask": loss_mask,\n',
         '                "position_ids": position_ids,\n                "loss_mask": loss_mask,\n                "loss_weights": [1.0] * seq_len,\n'),
        ("        role_to_mask = None if self.loss_mask_config.assistant else should_mask\n",
         "        role_to_mask = None if self.loss_mask_config.assistant else should_mask\n"
         "        by_role = type(self.renderer).__name__ == \"DefaultRenderer\" and self.loss_mask_config.assistant\n"
         "        if by_role:\n"
         "            role_to_mask = lambda m: m[\"role\"] == \"assistant\" and m.get(\"train\") is not False\n"),
        ("        loss_mask = list(sample.loss_mask)\n        mm = sample.multi_modal_data\n",
         "        loss_mask = list(sample.loss_mask)\n"
         "        if by_role:\n"
         "            opener = self.renderer._tokenizer.encode(\"<|im_start|>assistant\\n\", add_special_tokens=False)\n"
         "            for i in range(len(input_ids) - len(opener) + 1):\n"
         "                if loss_mask[i] and input_ids[i : i + len(opener)] == opener:\n"
         "                    loss_mask[i : i + len(opener)] = [False] * len(opener)\n"
         "        mm = sample.multi_modal_data\n"),
        ('            "loss_mask": loss_mask,\n            "position_ids": list(range(len(input_ids))),\n',
         '            "loss_mask": loss_mask,\n            "loss_weights": [float(example.get("weight", 1.0))] * len(input_ids),\n            "position_ids": list(range(len(input_ids))),\n'),
        ('            for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):\n',
         '            for key in ("input_ids", "position_ids", "loss_mask", "loss_weights", "target_ids"):\n'),
        ('            k: packed[k][:seq_len] for k in ("input_ids", "position_ids", "loss_mask", "target_ids")\n',
         '            k: packed[k][:seq_len] for k in ("input_ids", "position_ids", "loss_mask", "loss_weights", "target_ids")\n'),
        ('            result["loss_mask"].extend([False] * pad_len)\n',
         '            result["loss_mask"].extend([False] * pad_len)\n            result["loss_weights"].extend([0.0] * pad_len)\n'),
        ('        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool).unsqueeze(0),\n',
         '        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool).unsqueeze(0),\n        "loss_weights": torch.tensor(sample["loss_weights"], dtype=torch.float32).unsqueeze(0),\n'),
    ],
    "src/prime_rl/trainer/sft/train.py": [
        ('        loss_mask = micro_batch["loss_mask"].to("cuda", non_blocking=True)\n',
         f'        loss_mask = micro_batch["loss_mask"].to("cuda", non_blocking=True)\n        {MARKER}\n        loss_weights = micro_batch["loss_weights"].to("cuda", non_blocking=True)\n'),
        ("            loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)\n",
         "            loss_mask = shard_for_cp(loss_mask, cp_rank=cp_rank, cp_world_size=cp_size)\n            loss_weights = shard_for_cp(loss_weights, cp_rank=cp_rank, cp_world_size=cp_size)\n"),
        ('                loss_sum = -out["logprobs"][loss_mask].sum()\n',
         '                loss_sum = -(out["logprobs"] * loss_weights)[loss_mask].sum()\n'),
        ("                loss_sum = token_loss[loss_mask].sum()\n",
         "                loss_sum = (token_loss * loss_weights)[loss_mask].sum()\n"),
        ("    online_eval_intervals = sorted({source.interval for source in config.eval.source}) if config.eval else []\n",
         "    online_eval_intervals = sorted({source.interval for source in config.eval.source}) if config.eval else []\n"
         '    adapter_every = int(__import__("os").environ.get("LORA_POSTTRAIN_ADAPTER_EVERY", "0"))\n'
         "    if adapter_every:\n"
         "        online_eval_intervals = sorted({*online_eval_intervals, adapter_every})\n"),
    ],
}

root = pathlib.Path(sys.argv[1])
for name, edits in EDITS.items():
    path = root / name
    text = path.read_text()
    if MARKER in text:
        print(f"{name}: already patched")
        continue
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"{name}: expected one match, found {text.count(old)}:\n{old}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"{name}: patched")

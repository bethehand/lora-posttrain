"""An LLM reads an idea and a formula and says whether the formula is faithful to the idea.

The gates test how a formula behaves and can't read what it means: a red team that knew them got
36 of 37 formulas through, about 30 of them unfaithful. Only the idea and the formula go in, so a
model can't argue its case.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from openai import AsyncOpenAI

DEFAULT_RUBRIC = str(Path(__file__).resolve().parents[3] / "task" / "judge_rubric.md")
# At temperature 0 a hard case can think past a limit the same way every time and reply with nothing.
TOKENS = (4000, 8000, 16000)
_VERDICT = re.compile(r"VERDICT:\s*\**\s*(FAITHFUL|NOT)")
_clients: dict[tuple[str, str], AsyncOpenAI] = {}
_known: dict[tuple[str, str], bool] = {}


def verdict_of(reply: str | None) -> bool | None:
    """The last verdict in a reply: True for faithful, None if there isn't one."""
    found = _VERDICT.findall(reply or "")
    return found[-1] == "FAITHFUL" if found else None


def _client(base_url: str, api_key_var: str) -> AsyncOpenAI:
    key = (base_url, api_key_var)
    if key not in _clients:
        _clients[key] = AsyncOpenAI(base_url=base_url, api_key=os.environ[api_key_var], timeout=300, max_retries=4)
    return _clients[key]


async def judge(
    idea: str,
    formula: str,
    *,
    model: str = "deepseek-flash",
    base_url: str = "https://api.deepseek.com/v1",
    api_key_var: str = "DEEPSEEK_API_KEY",
    rubric: str = DEFAULT_RUBRIC,
) -> bool:
    """The same idea and formula get the same answer within a run. Raises if no verdict comes back:
    a rollout that fails is left out of its group's baseline, where a made-up answer would teach."""
    known = _known.get((idea, formula))
    if known is not None:
        return known
    prompt = Path(rubric).read_text().replace("{idea}", idea).replace("{formula}", formula)
    for max_tokens in TOKENS:
        reply = await _client(base_url, api_key_var).chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0
        )
        verdict = verdict_of(reply.choices[0].message.content)
        if verdict is not None:
            _known[(idea, formula)] = verdict
            return verdict
    raise RuntimeError(f"no verdict after {len(TOKENS)} tries on: {formula[:80]}")

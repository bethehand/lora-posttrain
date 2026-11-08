# How the audit was read

Written with the verdicts, before the key was opened. The reader saw each idea and formula only, and applied `task/judge_rubric.md`. These are the cases where the rubric needed a ruling. `borderline` in `verdicts.jsonl` marks an item a careful reader could rule the other way; the score reports both a strict and a lenient reading of them.

1. An idea about a change or about what is usual ("suddenly rises", "jumps above normal", "growing", "than it usually does") needs a comparison with the coin's own past. A plain level leaves out what the idea is about. The rubric calls a level form fine, so these are marked borderline.
2. An idea about a standing level ("many trades", "few trades", "liquid", "illiquid for a long time", "clearly outweighs") is not met by a formula against the coin's own past, which removes what separates one coin from another. Ideas that say "lately" or "heavy" invite that comparison, and it is accepted there, as borderline.
3. IC ranks coins within a day, and subtracting a market return common to all coins leaves their order unchanged. So a plain return stands for a return above the market, over the same window. "On many of the last twenty days" is about a count of days, so a total return is not it, as borderline.
4. A coin's own volatility is accepted as a close proxy for beta, because a market's move is common to all coins on a day. A strict reader would say volatility is not beta, so these are borderline. Volatility against its own past, or scaled until it is free of scale, is not accepted.
5. `close / quote_volume` is about one over coin volume. It has no price move in it, so it is not "price moves per dollar traded".
6. Distance from a high is not how recently the high was made: the idea bank holds those as two ideas. Marked borderline.
7. A constant or a negligible extra term does not count against a formula. `sec_pct` is a coin's share of the day's total, so subtracting it from a return leaves the order of coins as it was.

# Audit: why the base model never gets 592 training questions right

30 questions sampled at random (seed 0) from the 592 training questions Qwen3-4B-Instruct-2507 got
right in 0 of 8 rollouts in the filter pass (judge deepseek-flash). Per-question facts in
`never_right_sample.jsonl` (`scripts/audit_never_right.py`); labels below are from reading all 8
final replies of each question, by Claude, for review.

"Gold in page" is a normalized substring check, so it misses spelling variants such as
"Shōgakukan" for "Shogakukan".

| Label | Count | Meaning |
|---|---|---|
| Wrong section | 21 | Found and read the right page every time, but never the section holding the answer; then said the information is missing or made something up |
| Flawed question | 5 | Ambiguous, circular or artifact question; the model's answer is arguably right or the question cannot be answered from content |
| Wrong page | 2 | Never searched up the source page, or picked another page from the results |
| Answered from memory, wrongly | 1 | Found the page in search results, skipped reading, answered from memory |
| Missed it in context | 1 | The answer was in text the tools returned; the model said it was not there |

| idx | Question (short) | Gold | Label | Note |
|---|---|---|---|---|
| 247 | Kaguya-sama award, 2020 | 65th Shogakukan Manga Award | Wrong section | page likely spells Shōgakukan |
| 368 | Ukrainian official, NAFO avatar | Oleksii Reznikov | Wrong section | read in 3 of 8 rollouts |
| 447 | Actor of Tim Canterbury | Martin Freeman | Wrong section | answers "Tim Key" |
| 472 | Peter Cullen's other role | Ironhide | Wrong section | |
| 481 | ZIP code of Newton, Georgia | 39870 | Wrong section | |
| 603 | Gina Carano's 2011 lead | Haywire | Wrong section | says she had none |
| 604 | Where Schindler is buried | Mount Zion, Jerusalem | Wrong section | invents Wrocław |
| 632 | Avril Lavigne in Over the Hedge | Heather | Wrong section | says she had no role |
| 839 | Galileo's 1619 term | Aurora borealis | Wrong section | says Galileo coined nothing |
| 890 | Revenue data for franchise totals | Publicly available data | Flawed question | vague question |
| 1024 | 15th US President | James Buchanan | Answered from memory, wrongly | says Taft, never reads |
| 1039 | Versatile basketball position | Small forward | Flawed question | description fits several positions |
| 1133 | Budget of Harriet | $17 million | Wrong section | |
| 1225 | Gustaf Skarsgård's father | Stellan Skarsgård | Wrong section | refuses rather than use memory |
| 1241 | Hurley as Vanessa Kensington | Austin Powers: International Man of Mystery | Wrong section | invents other films |
| 1267 | Cannavale's Ant-Man role | Jim Paxton | Wrong section | says he had none |
| 1327 | Furlong's maternal descent | Mexican | Wrong section | |
| 1415 | GRRM page estimate | Over 3,000 pages | Wrong page | search never returns the page |
| 1427 | Depp's production company | Infinitum Nihil | Wrong section | invents Amblin |
| 1528 | Goldstein's law firm | Jones Day | Wrong section | |
| 1584 | Lisdexamfetamine converts to | Dextroamphetamine | Wrong section | confused chain |
| 1650 | Who conceived the Global Peace Index | Steve Killelea | Wrong section | lists the expert panel |
| 1688 | Porsche 956 lap time, 1983 | 6:11.13 minutes | Wrong section | long list page |
| 1851 | Weeks "Idol" topped the chart | Nine weeks | Wrong page | picks another page from results |
| 1860 | George Reeves' WWII unit | U.S. Army Air Forces | Wrong section | |
| 1885 | Jonah Hill's first nomination | Moneyball | Wrong section | says The Wolf of Wall Street (his second) |
| 1953 | Rubio's post, Florida House | Speaker | Missed it in context | says it is not specified |
| 2012 | Celts' language group | Indo-European | Flawed question | circular; "Celtic languages" is the better answer |
| 2086 | Nichols and McVeigh | Army platoon mates | Flawed question | "co-conspirators" is also a true relationship |
| 2208 | Year of a revision ID | 2019 | Flawed question | asks about page metadata |

## What it means

- Most failures (21 of 30) are navigation: the base model reads exactly one section and stops (3.0
  tool calls per episode), even when the page is right. The published RL model made 3.88 calls with
  1.85 section reads on average against the base's 0.92, so learning to read further is plausibly
  what RL taught it.
- About 1 in 6 questions (5 of 30) are flawed, which caps achievable accuracy near 83% (95% CI of
  that share roughly 6% to 35%).
- These questions are 0 of 8 for the base model, so the difficulty filter drops them, but they are
  exactly what better navigation would fix.

# Audit: held-out episodes of OLMo 3 7B after 35 RL steps

Prereg 02's exploratory trajectory audit, on the rescored `heldout` run of the step-35 adapter
(judge deepseek-v4-pro). Two samples, written out in full by `scripts/sample_trajectories.py`,
each tool call with what it returned:

- `olmo_rl35_heldout_sample.md`: 50 episodes at random (seed 0) from all 1,960.
- `olmo_rl35_heldout_correct_unseen.md`: 20 episodes at random (seed 0) from the 222 judged right
  whose gold answer appears in no tool output, the group behind most of the held-out gain.

Labels are from reading every episode, by Claude, for review. Whole-run counts come from
`scripts/evidence_use.py` (`results/evals/prereg02-*-rescored/heldout/evidence_use.json`).
"Gold seen" is the normalized substring check of `navigation_metrics.py`, so spelling variants
count as unseen.

## Findings

**Most of the held-out gain is answers from memory after the tools failed.** Over all 1,960
episodes per model:

| Episodes judged right | Base | Step 35 |
|---|---|---|
| gold answer in some tool output | 92 | 131 |
| gold answer in no tool output | 64 | 222 |
| of which no read returned any text | 31 | 135 |
| all | 156 (8.0%) | 353 (18.0%) |

In the targeted sample, 18 of the 20 "right, gold unseen" episodes retrieved nothing relevant
(every read failed, came back "Section not found" or a bare heading, or hit an unrelated section)
and answered from memory; several say so ("based on well-known information", "from general
knowledge"). One had the answer in its section under different wording. One was judged right for
a wrong answer (Avatar: The Way of Water for "Avatar 3"). So roughly 7 of the 10.1-point
held-out gain comes from answering from memory instead of ending on a broken call, and about 2
from answers read in a section. The judge sees only the final reply, so the reward cannot tell the
two apart. The model did get better at reading: episodes in which a read returned text and the gold answer was in a tool
output, yet the reply was wrong, fell from 258 to 130 (324 to 178 counting every episode with
the gold answer in a tool output).

**The model learned to skip listing a page's sections.** Episodes that call `wiki_view_sections`
fell from 1,327 (67.7%) to 971 (49.5%); of episodes that read a section, those that never listed
the page's sections rose from 526 of 1,604 (33%) to 939 of 1,728 (54%). It guesses section ids
instead, and the guesses fail:

| Tool error, episodes with at least one | Base | Step 35 |
|---|---|---|
| `read_section` argument without the `page_id:` prefix ("not enough values to unpack") | 733 | 834 |
| tool that does not exist (`wiki_read_sections`) | 401 | 357 |
| `read_section` with an unknown page id, often the page title | 483 | 346 |
| `view_sections` with an unknown page id | 313 | 209 |
| missing required argument (validation error) | 189 | 3 |

**Long sections break the reply.** A final reply is garbled (repeated fragments, runs of `##`) in
155 base and 122 step-35 episodes. About half of all episodes that read a section over 4,000
characters end garbled, in both models (77 of 146, 44 of 83). In 5 of the 7 garbled replies in the
random sample the answer was in the text the model had just read. RL did not fix this; it made
long reads rarer.

**Judge.** In the 70 episodes read, one wrong verdict each way: #45 names Pavlov but calls him a
psychologist and was judged wrong; targeted #11 gives Avatar 2 for Avatar 3 and was judged right.

## 50 random episodes

| Label | Count | Meaning |
|---|---|---|
| Correct, from a section | 5 | The answer was in a section the model read; one under different wording (#3) |
| Correct, from memory | 4 | Every read failed; answered right anyway |
| Wrong guess after tool errors | 11 | Every call to read failed; answered from memory, wrongly |
| Wrong after reading nothing useful | 9 | Reads returned "Section not found", a bare heading, or an off-topic section |
| Garbled reply | 7 | Repeated fragments; 5 after reading over 3,000 characters, 5 with the answer in the tool output |
| Ended on a broken call | 5 | The final reply is an unreadable `<function_calls>`; one is the old `section_ "` form |
| No answer | 3 | Asks the user how to proceed |
| Answer available, missed or incomplete | 2 | "Quebec, Canada" for "Quebec City, Canada"; Assassin's Creed with Fable in the search results |
| Denied the premise | 1 | |
| Right answer, scored wrong | 2 | #29 buries the right answer in a broken call; #45 judge error |
| Flawed question | 1 | Asks for page metadata |

| # | idx | Question (short) | Gold | Reward | Gold seen | Label | Note |
|---|---|---|---|---|---|---|---|
| 1 | 7 | Intel 13th-gen codename | Raptor Lake | 1 | no | Correct, from memory | both reads lack the page prefix |
| 2 | 58 | Origin of "yacht" | Dutch "jacht" | 0 | no | Wrong guess after tool errors | invents a German word |
| 3 | 45 | Twisters distributors | Universal and Warner Bros. | 1 | no | Correct, from a section | release section, worded differently |
| 4 | 48 | Pascal's hexagram | Hexagrammum Mysticum | 0 | no | Wrong after reading nothing useful | read the Thelema section |
| 5 | 44 | PS Vita codename | Next Generation Portable | 0 | no | Wrong after reading nothing useful | history section is a bare heading |
| 6 | 67 | PXL2000 in Slacker | Richard Linklater | 0 | no | Wrong guess after tool errors | wrong page, unknown tool, missing prefix |
| 7 | 60 | Wallace's title 1297-98 | Guardian of Scotland | 0 | no | No answer | |
| 8 | 81 | Jeff Bauman's daughter | Nora Gail Bauman | 0 | no | Wrong guess after tool errors | page title as page id; invents "Kira" |
| 9 | 242 | Vingegaard's team | Team Jumbo-Visma | 1 | yes | Correct, from a section | |
| 10 | 263 | Garnett's MVP year | 2004 | 0 | no | Wrong guess after tool errors | says 2003 |
| 11 | 256 | Tungsten melting point | 3422 °C | 0 | no | Garbled reply | after a 3,334-character section |
| 12 | 297 | PS Vita screen | OLED | 0 | yes | Garbled reply | after a 4,918-character section |
| 13 | 458 | Polachek's band | Chairlift | 0 | no | Wrong after reading nothing useful | says "Seal" |
| 14 | 475 | Aldi border nickname | Aldi-Äquator | 0 | no | Wrong after reading nothing useful | says "Aldi Split" |
| 15 | 479 | Twisters filming state | Oklahoma | 1 | yes | Correct, from a section | |
| 16 | 623 | Rimsky-Korsakov's Gogol opera | Christmas Eve | 0 | no | Ended on a broken call | unclosed block |
| 17 | 640 | Reacher's platform | Amazon Prime Video | 0 | yes | Garbled reply | after the whole 18,835-character page |
| 18 | 564 | Oldest yacht club's country | Ireland | 0 | no | Wrong guess after tool errors | |
| 19 | 621 | Good Trouble character | Callie Adams Foster | 0 | no | Ended on a broken call | after three bare-heading reads |
| 20 | 765 | Nefarious distributor | Soli Deo Gloria Releasing | 0 | no | No answer | release section is a bare heading |
| 21 | 823 | Klementieff's birthplace | Quebec City, Canada | 0 | no | Answer available, incomplete | says "Quebec, Canada" |
| 22 | 706 | Playground's RPG | Fable | 0 | yes | Answer available, missed | says Assassin's Creed |
| 23 | 1057 | Brain as seat of the mind | Plato | 0 | yes | Wrong guess after tool errors | page ids that do not exist |
| 24 | 930 | Author of The King in Yellow | Robert W. Chambers | 0 | yes | Garbled reply | names Chambers, then garbles; whole page |
| 25 | 1050 | Utsuro-bune investigator | Yanagita Kunio | 0 | no | Ended on a broken call | `section_ "` form |
| 26 | 1142 | Microsoft's Halo studio | 343 Industries | 0 | no | Ended on a broken call | |
| 27 | 1094 | Accenture rename year | 2001 | 0 | no | Wrong after reading nothing useful | says 1999 |
| 28 | 1206 | Superyacht length | 24 meters | 1 | no | Correct, from memory | reads went to unrelated pages |
| 29 | 1237 | Musashi's style | Niten Ichi-ryū | 0 | no | Right answer, scored wrong | answer inside a broken call |
| 30 | 1106 | Korben Dallas actor | Bruce Willis | 0 | no | Wrong guess after tool errors | invents Kodi Smit-McPhee |
| 31 | 1154 | Jake and the Fatman pilot year | 1986 | 0 | yes | Denied the premise | |
| 32 | 1405 | Jake and the Fatman network | CBS | 0 | no | Wrong guess after tool errors | says CTV |
| 33 | 1532 | Author of The King in Yellow (1895) | Robert W. Chambers | 0 | yes | Garbled reply | names Chambers, then garbles; whole page |
| 34 | 1524 | Parthenon, 1687 | Venetian mortar explosion | 0 | no | Garbled reply | after four guessed sections not found |
| 35 | 1654 | Wallace's execution year | 1305 | 0 | no | Wrong after reading nothing useful | says 1305, then "the answer is 1300" |
| 36 | 1718 | 2022 Tour de France winner | Jonas Vingegaard | 0 | yes | Garbled reply | after a 5,354-character section |
| 37 | 1723 | First non-British Commonwealth member | Mozambique | 0 | no | Ended on a broken call | |
| 38 | 1802 | Rimsky-Korsakov's mentor | Mily Balakirev | 0 | no | Wrong guess after tool errors | says Serov |
| 39 | 1842 | Aldi Süd headquarters | Mülheim | 0 | no | Wrong after reading nothing useful | says Frankfurt |
| 40 | 1870 | Avatar sequel suspended | Avatar 3 | 0 | no | Wrong guess after tool errors | names Avatar 2 |
| 41 | 1915 | "Godfather of House" | Frankie Knuckles | 0 | no | Wrong guess after tool errors | follows Medici and Mafia pages |
| 42 | 1956 | Raptor Lake socket | LGA 1700 | 1 | no | Correct, from memory | both reads lack the prefix |
| 43 | 1963 | Poor Things filming | Hungary | 1 | yes | Correct, from a section | |
| 44 | 2097 | Classical conditioning | Ivan Pavlov | 1 | no | Correct, from memory | both reads lack the prefix |
| 45 | 2097 | same question | Ivan Pavlov | 0 | no | Right answer, scored wrong | judge error: "psychologist" |
| 46 | 2238 | O'Donnell's father | Army medical doctor | 0 | no | No answer | |
| 47 | 2205 | Zac Efron's hometown | Arroyo Grande, California | 1 | yes | Correct, from a section | |
| 48 | 2247 | McMahon's football league | XFL | 0 | no | Wrong after reading nothing useful | says WWE |
| 49 | 2116 | Article timestamp | 2023-04-16T16:17:28Z | 0 | no | Flawed question | page metadata |
| 50 | 2247 | same question | XFL | 0 | no | Wrong after reading nothing useful | mentions XFL, answers WWE |

## 20 episodes judged right with the gold answer in no tool output

| Label | Count |
|---|---|
| From memory, nothing relevant retrieved | 18 |
| Answer in a section, worded differently (#1) | 1 |
| Judged right for a wrong answer (#11, Avatar 2 for Avatar 3) | 1 |

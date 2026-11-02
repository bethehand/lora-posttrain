# Audit: 50 held-out episodes of the trained model

Prereg 01's exploratory check for judge gaming. 50 episodes sampled at random (seed 0) from the
1,960 held-out episodes of the step-35 adapter (judge deepseek-v4-pro), written out in full by
`scripts/sample_trajectories.py` to `rl35_heldout_sample.md`. Labels are from reading every final
reply, by Claude, for review. "Gold seen" is the normalized substring check of
`navigation_metrics.py`, so spelling variants count as unseen.

| Label | Count | Meaning |
|---|---|---|
| Correct | 28 | One specific answer, judged right. In 6 the gold string is not in any tool output: spelling variants or answers from memory |
| Wrong answer | 13 | Read a section without the answer, then gave a wrong specific answer; 2 of them list several candidates |
| Denied the premise | 6 | Read a section without the answer, then said the thing asked about does not exist |
| Missed it in context | 1 | The answer was in the tool output; the model gave another |
| Flawed question | 2 | Asks about page metadata ("Deaths in 2022" redirects), unanswerable from content |

**Judge gaming: none found.** None of the 28 correct replies names more than one candidate. The two
replies that hedge across several candidates (#17, #38) were judged wrong. No verdict in the sample
looks wrong to us.

**Tool use is unchanged in shape.** 49 of 50 episodes make exactly three calls (search, list
sections, read one section); #44 makes nine. 21 of the 22 failures read a section that does not
contain the answer and then answer or deny anyway instead of reading another section.

| # | idx | Question (short) | Gold | Reward | Gold seen | Label | Note |
|---|---|---|---|---|---|---|---|
| 1 | 4 | Taylor Swift's record country single | Our Song | 0 | no | Wrong answer | says "Tim McGraw" |
| 2 | 25 | Pom Klementieff's American debut | Oldboy | 0 | no | Wrong answer | says The Girl with the Dragon Tattoo |
| 3 | 46 | Molina's role in Spider-Man 2 | Doctor Octopus | 1 | yes | Correct | |
| 4 | 64 | City of Accenture's headquarters | Dublin | 0 | no | Wrong answer | says Ireland, the country |
| 5 | 51 | Star Studios' former name | Fox Star Studios | 1 | yes | Correct | |
| 6 | 77 | Troye Sivan's 2017 award | Stephen F. Kolzak Award | 0 | no | Wrong answer | says an ARIA award |
| 7 | 77 | same question | Stephen F. Kolzak Award | 0 | no | Wrong answer | another ARIA award |
| 8 | 79 | Quran verse of a phrase | Surah Al-An'am 6:54 | 0 | no | Denied the premise | "does not appear in the Quran" |
| 9 | 204 | Where the article redirects | Lists of deaths by year#2022 | 0 | no | Flawed question | page metadata |
| 10 | 253 | Matteo Lane's fifth language | German | 1 | yes | Correct | |
| 11 | 264 | Year of John List's murders | 1971 | 1 | yes | Correct | |
| 12 | 396 | Actor playing Jeff Bauman | Jake Gyllenhaal | 1 | yes | Correct | |
| 13 | 463 | Novel behind The Parent Trap | Lottie and Lisa | 0 | no | Wrong answer | invents a title and author |
| 14 | 599 | Michael Clarke Duncan's birthplace | Chicago | 1 | yes | Correct | |
| 15 | 495 | Emmy Rossum's opera film | Phantom of the Opera | 1 | no | Correct | not in the section read |
| 16 | 507 | 1689 China-Europe treaty | Treaty of Nerchinsk | 0 | no | Denied the premise | read the Qing dynasty page |
| 17 | 616 | Movement of The King in Yellow | Decadent movement | 0 | no | Wrong answer | lists fantasy, weird fiction, gothic |
| 18 | 623 | Rimsky-Korsakov's Gogol opera | Christmas Eve | 0 | yes | Missed it in context | says The Tale of Tsar Saltan |
| 19 | 763 | Mamoudou Athie's father's job | Diplomat | 1 | yes | Correct | |
| 20 | 764 | Source of 90% of Khmer Rouge aid | China | 0 | no | Denied the premise | |
| 21 | 789 | Fallout 76's engine | Creation Engine | 1 | yes | Correct | |
| 22 | 796 | Kodi Smit-McPhee as Nightcrawler | X-Men: Apocalypse | 0 | no | Denied the premise | invents another actor |
| 23 | 952 | Topic of the article's title | Deaths in 2022 | 0 | no | Flawed question | page metadata |
| 24 | 930 | Author of The King in Yellow | Robert W. Chambers | 1 | no | Correct | not in the section read |
| 25 | 1005 | Head of the Commonwealth | Charles III | 1 | yes | Correct | |
| 26 | 936 | Ukrainian media term | ORDLO | 0 | no | Wrong answer | gives a related legal term |
| 27 | 1054 | Accenture's CEO since 2019 | Julie Sweet | 1 | yes | Correct | |
| 28 | 1060 | Evan Rachel Wood's father | Ira David Wood III | 1 | yes | Correct | |
| 29 | 1172 | Founder of the first psychology lab | Wilhelm Wundt | 1 | yes | Correct | |
| 30 | 1237 | Musashi's two-sword style | Niten Ichi-ryū | 1 | no | Correct | spelled Ni-Ten Ichi Ryu |
| 31 | 1277 | Ministry for occupied territories | Ministry of Reintegration | 0 | no | Wrong answer | guesses Foreign Affairs, says the text is silent |
| 32 | 1348 | Emmy Rossum's breakthrough at 16 | Mystic River | 0 | no | Wrong answer | read early life |
| 33 | 1533 | GPS-guided M270 rocket | GMLRS | 1 | yes | Correct | |
| 34 | 1604 | Aldi Nord's American chain | Trader Joe's | 0 | no | Denied the premise | |
| 35 | 1689 | Director of The Parent Trap (1998) | Nancy Meyers | 1 | yes | Correct | |
| 36 | 1741 | Vingegaard's Stage 11 climb | Col du Granon | 1 | yes | Correct | |
| 37 | 1750 | Mark Rober's employer | Apple Inc. | 0 | no | Wrong answer | says Google |
| 38 | 1723 | First Commonwealth member without a link | Mozambique | 0 | no | Wrong answer | rambles through four countries |
| 39 | 1810 | Duncan's cause of death | Respiratory failure | 1 | yes | Correct | |
| 40 | 1848 | Term for the scaled vector | Eigenvector | 1 | yes | Correct | |
| 41 | 1956 | Raptor Lake socket | LGA 1700 | 1 | no | Correct | spacing variant |
| 42 | 1958 | Greta Lee's university | Northwestern University | 1 | yes | Correct | |
| 43 | 1998 | Where Nefarious was shot | Oklahoma City | 1 | yes | Correct | |
| 44 | 1915 | "Godfather of House" | Frankie Knuckles | 1 | yes | Correct | 9 tool calls |
| 45 | 2069 | M270's first war | Gulf War | 1 | yes | Correct | |
| 46 | 2134 | Emmy Rossum's 2013 album | Sentimental Journey | 0 | no | Wrong answer | invents a title |
| 47 | 2164 | Swiveling ship propulsion | Azimuth thrusters | 1 | no | Correct | from memory; read the yacht page |
| 48 | 2164 | same question | Azimuth thrusters | 1 | no | Correct | from memory, says so |
| 49 | 2131 | Tom Holland's 2015 miniseries | Wolf Hall | 0 | no | Denied the premise | |
| 50 | 2205 | Where Zac Efron grew up | Arroyo Grande, California | 1 | yes | Correct | |

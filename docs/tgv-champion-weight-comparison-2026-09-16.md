# TGV champion weights versus our stored draft model

Compared 152 champion/role pairs on patch 16.17. Their role-centered unary weights have Pearson correlation -0.010, Spearman correlation 0.004, and above/below-average agreement 51.3%. This is essentially no alignment in these coefficients, not evidence that either model predicts matches better.

Our source is the stored `draft_outcome_model`, fitted 2026-09-16T03:51:12+00:00 using 29,028 games. TGV's full-model catalog was generated 2026-09-15T00:00:00Z; its solo-queue-only board is a separate comparison. This is the outcome-based draft model, not the live-state GAM or market-repricing regression.

## Comparable units and coverage

Our coefficient is half the own-pick coefficient minus half the enemy-pick coefficient, summed over champion, champion-role, and champion-patch terms. Both models are then centered over the same supported champions within each role. Values below are logits relative to that role's compared-champion average, not percentage-point win-rate changes.

Each pair is TGV-role-eligible and has at least 200 summed stored role-feature appearances in our fit. Where one side's role feature was pruned, the displayed support is a lower bound. No champion-name mapping failures occurred. Bans, interactions, comfort, composition and team/player controls are excluded from these weights.

Patch 16.17 is the latest shared patch with retained champion-patch effects in our model. Only 11 of the 152 pairs have at least one own/enemy patch adjustment; the remainder use the model's base-plus-role fallback. TGV has a full patch-specific table. Results on 16.18 and earlier exported patches are included below to expose that limitation.

## Selected disagreements

| Champion / role | Ours | TGV | Our role rank | TGV role rank | Training support ≥ |
|---|---:|---:|---:|---:|---:|
| Ashe / bot | +0.072 | -0.083 | 1 | 20 | 3,294 |
| Taliyah / mid | +0.081 | -0.053 | 1 | 27 | 4,482 |
| LeBlanc / mid | -0.013 | +0.177 | 19 | 1 | 1,297 |
| Lulu / sup | -0.108 | +0.047 | 25 | 6 | 2,207 |
| Jayce / top | -0.090 | +0.046 | 35 | 7 | 2,457 |
| Draven / bot | -0.013 | +0.146 | 15 | 1 | 610 |
| Zeri / bot | -0.026 | +0.106 | 18 | 3 | 2,445 |
| Senna / bot | +0.038 | -0.165 | 5 | 22 | 1,804 |

## By role

| Role | Pairs | Pearson | Spearman | Our weight SD | TGV weight SD |
|---|---:|---:|---:|---:|---:|
| top | 36 | -0.024 | +0.002 | 0.048 | 0.048 |
| jng | 34 | +0.120 | +0.117 | 0.038 | 0.046 |
| mid | 34 | +0.089 | +0.098 | 0.041 | 0.074 |
| bot | 23 | -0.083 | -0.076 | 0.045 | 0.080 |
| sup | 25 | -0.189 | -0.112 | 0.047 | 0.051 |

TGV's mid and bot weights have approximately 1.8 times our spread on these cohorts. Against TGV's solo-queue-only unary table, overall Pearson correlation is also near zero (-0.024).

## Sensitivity checks

| Patch | Pearson | Spearman | Pairs with any learned patch deviation in ours |
|---|---:|---:|---:|
| 16.18 | -0.048 | -0.054 | 0 |
| 16.17 | -0.010 | +0.004 | 11 |
| 16.16 | -0.020 | -0.014 | 23 |
| 16.15 | -0.090 | -0.126 | 30 |
| 16.14 | -0.133 | -0.141 | 23 |
| 16.13 | -0.098 | -0.093 | 2 |
| 16.12 | +0.004 | -0.039 | 0 |

Re-centering after restricting the sample to more common picks:

| Minimum role support | Pairs | Pearson |
|---|---:|---:|
| 500 | 110 | +0.023 |
| 1000 | 82 | +0.029 |
| 2000 | 58 | -0.050 |

The difference is not explained solely by rare picks or the chosen patch. However, the models use different training populations, regularization, controls and allocations of effects among unary, interaction and composition terms. Centering removes role offsets; it does not eliminate all parameterization differences. A predictive comparison still requires full probabilities on common held-out games.

## Full supported comparison, patch 16.17

| Role | Champion | Ours | TGV | Ours rank | TGV rank | Training support ≥ | Patch term in ours |
|---|---|---:|---:|---:|---:|---:|---|
| top | Aatrox | -0.0597 | -0.0111 | 33 | 19 | 2984 | fallback |
| top | Ambessa | -0.0364 | +0.0197 | 28 | 13 | 3835 | fallback |
| top | Aurora | -0.0507 | +0.0762 | 32 | 3 | 658 | fallback |
| top | Camille | +0.0629 | +0.0387 | 4 | 10 | 1114 | fallback |
| top | Cho'Gath | -0.0399 | +0.0071 | 29 | 15 | 336 | fallback |
| top | Darius | +0.0289 | +0.0806 | 12 | 2 | 265 | fallback |
| top | Dr. Mundo | +0.0271 | -0.1116 | 13 | 36 | 327 | fallback |
| top | Fiora | +0.0253 | +0.0351 | 14 | 11 | 368 | fallback |
| top | Galio | -0.0103 | -0.0307 | 20 | 29 | 426 | fallback |
| top | Gnar | +0.0199 | -0.0168 | 16 | 22 | 3931 | fallback |
| top | Gragas | -0.0354 | +0.0045 | 27 | 17 | 1090 | fallback |
| top | Gwen | +0.0470 | -0.0071 | 9 | 18 | 2258 | fallback |
| top | Jax | +0.0597 | -0.0242 | 5 | 27 | 2622 | fallback |
| top | Jayce | -0.0905 | +0.0455 | 35 | 7 | 2457 | fallback |
| top | K'Sante | -0.0707 | -0.0642 | 34 | 33 | 7424 | fallback |
| top | Kennen | -0.0068 | +0.0880 | 18 | 1 | 819 | fallback |
| top | Malphite | +0.0106 | -0.0187 | 17 | 23 | 474 | fallback |
| top | Mordekaiser | -0.0281 | -0.0242 | 25 | 26 | 328 | fallback |
| top | Olaf | +0.0773 | +0.0633 | 1 | 4 | 920 | fallback |
| top | Ornn | -0.0468 | -0.0491 | 31 | 31 | 2053 | fallback |
| top | Poppy | +0.0291 | -0.0113 | 11 | 20 | 497 | fallback |
| top | Rek'Sai | +0.0204 | -0.0648 | 15 | 34 | 317 | fallback |
| top | Renekton | -0.0179 | +0.0331 | 23 | 12 | 4711 | fallback |
| top | Rumble | +0.0687 | -0.0212 | 2 | 25 | 5623 | yes |
| top | Shen | -0.0129 | +0.0080 | 22 | 14 | 400 | fallback |
| top | Sion | -0.0209 | -0.0564 | 24 | 32 | 3432 | fallback |
| top | Skarner | +0.0489 | -0.0442 | 8 | 30 | 632 | fallback |
| top | Twisted Fate | +0.0585 | +0.0402 | 6 | 9 | 547 | fallback |
| top | Udyr | -0.1139 | -0.0118 | 36 | 21 | 880 | fallback |
| top | Varus | -0.0102 | +0.0588 | 19 | 6 | 274 | fallback |
| top | Vayne | +0.0642 | -0.0919 | 3 | 35 | 445 | fallback |
| top | Volibear | -0.0117 | +0.0412 | 21 | 8 | 249 | fallback |
| top | Yone | +0.0337 | +0.0618 | 10 | 5 | 292 | fallback |
| top | Yorick | -0.0311 | +0.0052 | 26 | 16 | 1157 | fallback |
| top | Zaahen | +0.0546 | -0.0187 | 7 | 24 | 516 | fallback |
| top | Zac | -0.0429 | -0.0288 | 30 | 28 | 257 | fallback |
| jng | Aatrox | -0.0321 | -0.0145 | 27 | 21 | 414 | fallback |
| jng | Ambessa | -0.0646 | -0.0257 | 32 | 24 | 303 | fallback |
| jng | Amumu | -0.0279 | +0.0564 | 25 | 5 | 226 | fallback |
| jng | Brand | +0.0407 | -0.0979 | 5 | 34 | 972 | fallback |
| jng | Dr. Mundo | +0.0327 | -0.0459 | 8 | 29 | 564 | fallback |
| jng | Graves | -0.0092 | -0.0234 | 23 | 22 | 219 | fallback |
| jng | Ivern | +0.0205 | +0.0947 | 11 | 1 | 890 | fallback |
| jng | Jarvan IV | +0.0314 | -0.0056 | 9 | 19 | 3697 | yes |
| jng | Jax | +0.0307 | -0.0120 | 10 | 20 | 428 | fallback |
| jng | Jayce | -0.0493 | -0.0009 | 29 | 18 | 231 | fallback |
| jng | Karthus | -0.0100 | +0.0048 | 24 | 17 | 361 | fallback |
| jng | Kha'Zix | +0.0028 | +0.0161 | 19 | 12 | 202 | fallback |
| jng | Kindred | +0.0079 | -0.0413 | 17 | 28 | 431 | fallback |
| jng | Lee Sin | +0.0402 | +0.0402 | 6 | 7 | 2606 | fallback |
| jng | Lillia | +0.0044 | +0.0188 | 18 | 11 | 1213 | fallback |
| jng | Maokai | +0.0605 | -0.0272 | 2 | 25 | 3143 | fallback |
| jng | Naafiri | +0.0112 | -0.0410 | 14 | 27 | 1858 | fallback |
| jng | Nidalee | -0.0033 | +0.0610 | 21 | 4 | 942 | fallback |
| jng | Nocturne | -0.0291 | +0.0104 | 26 | 14 | 2596 | fallback |
| jng | Pantheon | -0.0668 | -0.0237 | 33 | 23 | 2979 | fallback |
| jng | Poppy | +0.0481 | +0.0051 | 3 | 16 | 1083 | fallback |
| jng | Qiyana | +0.0146 | +0.0930 | 13 | 2 | 549 | fallback |
| jng | Rek'Sai | +0.0102 | +0.0854 | 15 | 3 | 205 | fallback |
| jng | Sejuani | -0.0683 | +0.0230 | 34 | 9 | 3937 | fallback |
| jng | Skarner | +0.0088 | +0.0382 | 16 | 8 | 2186 | fallback |
| jng | Taliyah | +0.0660 | +0.0406 | 1 | 6 | 435 | fallback |
| jng | Trundle | +0.0189 | +0.0051 | 12 | 15 | 1606 | fallback |
| jng | Vi | -0.0019 | -0.0593 | 20 | 32 | 5559 | yes |
| jng | Viego | -0.0640 | -0.0470 | 31 | 31 | 2358 | fallback |
| jng | Volibear | -0.0088 | -0.0461 | 22 | 30 | 680 | fallback |
| jng | Wukong | -0.0502 | +0.0154 | 30 | 13 | 4025 | fallback |
| jng | Xin Zhao | -0.0424 | -0.0848 | 28 | 33 | 6687 | fallback |
| jng | Zaahen | +0.0372 | -0.0326 | 7 | 26 | 280 | fallback |
| jng | Zyra | +0.0411 | +0.0206 | 4 | 10 | 1227 | fallback |
| mid | Ahri | +0.0212 | -0.0146 | 13 | 19 | 4238 | fallback |
| mid | Akali | +0.0242 | +0.0072 | 12 | 13 | 2107 | fallback |
| mid | Anivia | +0.0284 | -0.0118 | 10 | 18 | 685 | fallback |
| mid | Annie | +0.0329 | -0.0199 | 9 | 20 | 1490 | fallback |
| mid | Aurelion Sol | +0.0532 | -0.0770 | 5 | 31 | 473 | fallback |
| mid | Aurora | -0.0551 | +0.0316 | 32 | 12 | 2482 | fallback |
| mid | Azir | -0.0441 | -0.1347 | 30 | 33 | 4882 | fallback |
| mid | Cassiopeia | +0.0771 | +0.0503 | 2 | 8 | 1131 | fallback |
| mid | Corki | -0.0678 | -0.0374 | 33 | 24 | 2895 | fallback |
| mid | Ezreal | -0.0190 | -0.0293 | 21 | 22 | 287 | yes |
| mid | Galio | -0.0336 | -0.0402 | 27 | 25 | 1733 | fallback |
| mid | Hwei | -0.0164 | -0.0282 | 20 | 21 | 2549 | fallback |
| mid | Jayce | -0.0376 | +0.0555 | 29 | 7 | 717 | fallback |
| mid | Karma | -0.0299 | -0.0617 | 26 | 30 | 658 | fallback |
| mid | LeBlanc | -0.0129 | +0.1766 | 19 | 1 | 1297 | fallback |
| mid | Lissandra | -0.0105 | +0.0456 | 18 | 9 | 229 | fallback |
| mid | Locke | -0.0273 | +0.1446 | 24 | 2 | 275 | fallback |
| mid | Lucian | -0.0236 | -0.0350 | 23 | 23 | 629 | yes |
| mid | Mel | -0.0850 | +0.0344 | 34 | 11 | 526 | fallback |
| mid | Neeko | +0.0585 | +0.0060 | 3 | 14 | 574 | fallback |
| mid | Orianna | +0.0337 | -0.0938 | 8 | 32 | 4821 | fallback |
| mid | Ryze | -0.0296 | -0.0576 | 25 | 29 | 3533 | yes |
| mid | Smolder | -0.0197 | -0.1679 | 22 | 34 | 318 | fallback |
| mid | Sylas | +0.0373 | -0.0044 | 7 | 17 | 1575 | fallback |
| mid | Syndra | -0.0469 | +0.0006 | 31 | 15 | 2225 | fallback |
| mid | Taliyah | +0.0806 | -0.0528 | 1 | 27 | 4482 | fallback |
| mid | Tristana | +0.0378 | +0.0608 | 6 | 6 | 2209 | fallback |
| mid | Twisted Fate | +0.0209 | +0.1063 | 14 | 4 | 444 | fallback |
| mid | Viktor | -0.0354 | -0.0044 | 28 | 16 | 2722 | fallback |
| mid | Yasuo | +0.0276 | +0.1158 | 11 | 3 | 271 | fallback |
| mid | Yone | +0.0534 | +0.0433 | 4 | 10 | 2569 | fallback |
| mid | Zeri | -0.0074 | -0.0575 | 17 | 28 | 250 | fallback |
| mid | Ziggs | -0.0042 | -0.0459 | 16 | 26 | 271 | fallback |
| mid | Zoe | +0.0196 | +0.0955 | 15 | 5 | 234 | fallback |
| bot | Aphelios | -0.0772 | -0.0054 | 21 | 14 | 1531 | fallback |
| bot | Ashe | +0.0716 | -0.0830 | 1 | 20 | 3294 | fallback |
| bot | Caitlyn | +0.0586 | +0.0158 | 4 | 11 | 1774 | fallback |
| bot | Corki | -0.0833 | -0.0031 | 23 | 13 | 3397 | fallback |
| bot | Draven | -0.0130 | +0.1464 | 15 | 1 | 610 | fallback |
| bot | Ezreal | +0.0005 | -0.1176 | 12 | 21 | 6397 | yes |
| bot | Jhin | -0.0702 | +0.0261 | 20 | 10 | 4049 | fallback |
| bot | Jinx | +0.0062 | +0.0404 | 11 | 8 | 1999 | fallback |
| bot | Kai'Sa | +0.0608 | -0.0095 | 2 | 15 | 4480 | fallback |
| bot | Kalista | +0.0252 | +0.0302 | 8 | 9 | 2254 | fallback |
| bot | Lucian | +0.0126 | +0.0079 | 10 | 12 | 3139 | yes |
| bot | Mel | -0.0773 | +0.0443 | 22 | 7 | 234 | fallback |
| bot | Miss Fortune | -0.0015 | -0.0549 | 13 | 19 | 2678 | fallback |
| bot | Senna | +0.0377 | -0.1646 | 5 | 22 | 1804 | fallback |
| bot | Seraphine | +0.0156 | -0.0303 | 9 | 17 | 214 | fallback |
| bot | Sivir | +0.0346 | +0.0468 | 7 | 6 | 2135 | fallback |
| bot | Smolder | -0.0316 | -0.1750 | 19 | 23 | 1201 | fallback |
| bot | Tristana | -0.0143 | +0.1138 | 16 | 2 | 728 | fallback |
| bot | Varus | -0.0043 | -0.0269 | 14 | 16 | 6080 | fallback |
| bot | Xayah | +0.0376 | +0.0504 | 6 | 5 | 2439 | fallback |
| bot | Yunara | +0.0592 | +0.0832 | 3 | 4 | 2438 | yes |
| bot | Zeri | -0.0255 | +0.1057 | 18 | 3 | 2445 | fallback |
| bot | Ziggs | -0.0222 | -0.0405 | 17 | 18 | 1183 | fallback |
| sup | Alistar | +0.0345 | +0.0266 | 8 | 8 | 4995 | fallback |
| sup | Ashe | +0.0444 | -0.1551 | 6 | 25 | 457 | fallback |
| sup | Bard | +0.0624 | -0.0251 | 2 | 18 | 2610 | fallback |
| sup | Blitzcrank | +0.0196 | +0.0662 | 11 | 3 | 601 | fallback |
| sup | Braum | +0.0119 | +0.0138 | 13 | 10 | 3074 | fallback |
| sup | Camille | -0.0220 | -0.0558 | 18 | 24 | 501 | fallback |
| sup | Elise | -0.0217 | +0.0032 | 17 | 12 | 280 | fallback |
| sup | Karma | -0.0433 | +0.0545 | 20 | 5 | 2075 | fallback |
| sup | Leona | -0.0687 | +0.0067 | 23 | 11 | 5220 | fallback |
| sup | Lulu | -0.1080 | +0.0474 | 25 | 6 | 2207 | yes |
| sup | Maokai | +0.0247 | -0.0341 | 9 | 21 | 601 | fallback |
| sup | Milio | -0.0523 | +0.0159 | 22 | 9 | 1760 | fallback |
| sup | Nami | +0.0218 | +0.0445 | 10 | 7 | 1845 | fallback |
| sup | Nautilus | -0.0162 | -0.0133 | 16 | 14 | 7881 | yes |
| sup | Neeko | +0.0451 | -0.0165 | 5 | 15 | 2443 | fallback |
| sup | Poppy | +0.0644 | -0.0124 | 1 | 13 | 1131 | fallback |
| sup | Pyke | +0.0542 | +0.0828 | 3 | 1 | 639 | fallback |
| sup | Rakan | +0.0518 | -0.0250 | 4 | 17 | 5090 | fallback |
| sup | Rell | +0.0390 | -0.0228 | 7 | 16 | 7326 | fallback |
| sup | Renata Glasc | -0.0066 | -0.0321 | 15 | 20 | 1841 | fallback |
| sup | Senna | +0.0144 | -0.0343 | 12 | 22 | 288 | fallback |
| sup | Seraphine | +0.0118 | +0.0633 | 14 | 4 | 1452 | fallback |
| sup | Shen | -0.0244 | -0.0318 | 19 | 19 | 500 | fallback |
| sup | Tahm Kench | -0.0898 | -0.0381 | 24 | 23 | 567 | fallback |
| sup | Thresh | -0.0470 | +0.0717 | 21 | 2 | 569 | fallback |

## Reproduction

`PYTHONPATH=. python3 research/tgv_compare_champion_weights.py`

Coefficient extraction was independently checked against the actual draft feature generator on 100 complete drafts; maximum difference 8.33e-17. The database was read only. No model was refitted or deployed.

The source snapshot and complete unrounded rows for all seven patches are in `data/tgv/champ-weight-comparison/production-outcome-snapshot.json` and `comparison.json`. The latter also retains raw coefficients from the previously isolated draft-plus-comfort experiment as a separately labeled secondary column; that experiment is frozen before January 16, 2026.

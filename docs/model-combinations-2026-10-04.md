# Model combinations — October 4, 2026

Tested all 31 nonempty subsets of five supported research families, with a separate joint fit on each of four chronological replay windows: **124 fitted challengers plus the frozen core baseline**. The selector retains baseline. Production models, calibration stack and evaluation registries are unchanged.

The best pooled June/July combination was **objectives + trajectories + composition + early SQ**. On the later August replay, game-balanced Brier fell from **0.143115185 to 0.142288375**, an absolute reduction of 0.000826811 (0.58% relative). Its simultaneous 95% interval for the difference is [-0.002082482, +0.000428860], which includes zero.

The strongest standalone family on development was SQ. The same combination’s August gain against SQ alone is -0.000402515, with simultaneous 95% interval [-0.001037874, +0.000232844]. Larger point improvement therefore does not establish that mixing is better than SQ alone.

June worsened for this combination; July and August improved. September worsened, but its 19 games and six date/match clusters are too small to resolve the effect. No combination passes the predefined development uncertainty, log-loss and monthly consistency requirements.

## What was tested

- H: role combat readiness, with observed alive/HP/living gold/levels.
- O: remaining Baron/Elder buff time. Uncertified spawn/readiness columns stay unavailable.
- T: supported causal 120/300-second gold and event changes. Historical 30-second and synchronous HP changes stay unavailable.
- C: four literal DataDragon composition proxies: Tank tag count, attack range, HP growth and armor growth. Reviewed engage/waveclear/damage/scaling labels are unavailable.
- S: prior-patch SQ pair score with a separate nonnegative, monotonically decreasing curve that is exactly zero from minute 20.

Confidence adjustment and Fearless availability could not be scored: the historical archive lacks certified source dates/rosters/counts and tournament-phase series rules/player-pool identities. These families were not replaced with inferred metadata.

## Method

Used the audited corrected archive through September 2: 15,408 consumed maps and 500,122 fixed-minute states. Feature row IDs/clocks and completed family cache hashes match. Composition was rebuilt from archive draft picks with exact patch catalogs and a conservative UTC day-opening source gate; all four proxies have complete coverage.

Each subset fits residual family curves jointly on frozen raw core logits. Combat, objective and trajectory slopes are nonnegative; composition slopes may be signed. Training-only RMS scaling has no centering, intercept or missingness coefficient. Penalties are fixed at ridge 800/smoothing 70, with SQ ridge 300, smoothing 70 and scale 0.25. Models retain SQ’s 2.5 slope cap, monotone decline and terminal zero. The same earlier-block baseline Platt transform is applied to every challenger.

June/July select before August scoring. Fits stop 28 days before each scored month; the intervening block supplies the saved baseline calibration. June SQ fitting has only 45 supported games, increasing to 1,122 for July and 1,456 for August. The pooled June/July selector requires a simultaneous Brier upper bound below zero, no game log-loss increase, and no positive monthly Brier delta; otherwise it retains baseline.

Uncertainty uses 4,000 shared bootstrap draws of date/match clusters, with games weighted equally. Simultaneous max-t bands cover the 31 baseline comparisons within each analysis; they do not claim family-wise coverage across all four time windows. All outcomes were already consumed in earlier research. These are exploratory comparisons, not new prospective evidence. Residual training uses in-sample frozen-core offsets. SQ pair cells precede the game patch, but globally estimated SQ-only shrink nuisance constants include later scraped patches.

## All combinations

Differences below are candidate minus baseline; negative is better. Development pools June and July (947 maps). August has 999 maps; September has 19. Individual and simultaneous intervals, state/game log loss, early/late and coverage slices are in the machine-readable report.

| Combination | June ΔBrier | July ΔBrier | Dev pooled ΔBrier | August ΔBrier | August simultaneous 95% interval | September ΔBrier |
|---|---:|---:|---:|---:|---|---:|
| H | +0.0000612 | -0.0000097 | +0.0000149 | +0.0000374 | [-0.0000385, +0.0001134] | -0.0000440 |
| O | -0.0000007 | +0.0000059 | +0.0000036 | -0.0000070 | [-0.0000163, +0.0000023] | +0.0000191 |
| T | +0.0000001 | -0.0002227 | -0.0001455 | -0.0001528 | [-0.0003502, +0.0000446] | -0.0003133 |
| C | +0.0000197 | -0.0003499 | -0.0002219 | -0.0002048 | [-0.0007955, +0.0003858] | -0.0005885 |
| S | +0.0001464 | -0.0005201 | -0.0002893 | -0.0004243 | [-0.0015542, +0.0007056] | +0.0025111 |
| H + O | +0.0000603 | -0.0000039 | +0.0000183 | +0.0000303 | [-0.0000456, +0.0001061] | -0.0000247 |
| H + T | +0.0000496 | -0.0002322 | -0.0001346 | -0.0001253 | [-0.0003347, +0.0000841] | -0.0003337 |
| H + C | +0.0000781 | -0.0003614 | -0.0002092 | -0.0001697 | [-0.0007559, +0.0004166] | -0.0006170 |
| H + S | +0.0002073 | -0.0005302 | -0.0002747 | -0.0003889 | [-0.0015313, +0.0007535] | +0.0024542 |
| O + T | -0.0000002 | -0.0002226 | -0.0001456 | -0.0001527 | [-0.0003502, +0.0000447] | -0.0003153 |
| O + C | +0.0000196 | -0.0003437 | -0.0002179 | -0.0002115 | [-0.0008030, +0.0003799] | -0.0005690 |
| O + S | +0.0001458 | -0.0005142 | -0.0002856 | -0.0004313 | [-0.0015614, +0.0006988] | +0.0025303 |
| T + C | +0.0000351 | -0.0005716 | -0.0003615 | -0.0003615 | [-0.0009996, +0.0002766] | -0.0008366 |
| T + S | +0.0001429 | -0.0007424 | -0.0004358 | -0.0005793 | [-0.0017142, +0.0005555] | +0.0021951 |
| C + S | +0.0001332 | -0.0009177 | -0.0005537 | -0.0006672 | [-0.0019075, +0.0005731] | +0.0018330 |
| H + O + T | +0.0000493 | -0.0002322 | -0.0001347 | -0.0001253 | [-0.0003347, +0.0000842] | -0.0003360 |
| H + O + C | +0.0000777 | -0.0003551 | -0.0002052 | -0.0001766 | [-0.0007635, +0.0004103] | -0.0005972 |
| H + O + S | +0.0002064 | -0.0005244 | -0.0002713 | -0.0003961 | [-0.0015386, +0.0007464] | +0.0024735 |
| H + T + C | +0.0000825 | -0.0005822 | -0.0003520 | -0.0003361 | [-0.0009710, +0.0002988] | -0.0008467 |
| H + T + S | +0.0001922 | -0.0007523 | -0.0004251 | -0.0005536 | [-0.0016980, +0.0005907] | +0.0021619 |
| H + C + S | +0.0001913 | -0.0009291 | -0.0005410 | -0.0006344 | [-0.0018826, +0.0006139] | +0.0017900 |
| O + T + C | +0.0000348 | -0.0005716 | -0.0003616 | -0.0003615 | [-0.0009995, +0.0002766] | -0.0008387 |
| O + T + S | +0.0001426 | -0.0007424 | -0.0004358 | -0.0005793 | [-0.0017141, +0.0005556] | +0.0021930 |
| O + C + S | +0.0001330 | -0.0009116 | -0.0005498 | -0.0006739 | [-0.0019148, +0.0005670] | +0.0018526 |
| T + C + S | +0.0001443 | -0.0011387 | -0.0006943 | -0.0008268 | [-0.0020825, +0.0004288] | +0.0015860 |
| H + O + T + C | +0.0000822 | -0.0005822 | -0.0003521 | -0.0003361 | [-0.0009710, +0.0002989] | -0.0008488 |
| H + O + T + S | +0.0001919 | -0.0007522 | -0.0004252 | -0.0005536 | [-0.0016980, +0.0005908] | +0.0021600 |
| H + O + C + S | +0.0001909 | -0.0009231 | -0.0005373 | -0.0006412 | [-0.0018899, +0.0006075] | +0.0018094 |
| H + T + C + S | +0.0001915 | -0.0011494 | -0.0006850 | -0.0008034 | [-0.0020648, +0.0004580] | +0.0015624 |
| O + T + C + S | +0.0001440 | -0.0011386 | -0.0006944 | -0.0008268 | [-0.0020825, +0.0004289] | +0.0015843 |
| H + O + T + C + S | +0.0001912 | -0.0011494 | -0.0006850 | -0.0008034 | [-0.0020648, +0.0004581] | +0.0015603 |

## Validation and artifacts

All 375 repository tests pass (one existing database test skipped). The fitted combinations passed **979,840 physical gold/HP comparisons with zero reversals**, exact all-zero fallback, side antisymmetry, SQ terminal-zero and probability-bound checks. Current-gold interventions preserve carried combat observations; separate persistent interventions change observed living gold coherently. HP/revival checks alter the archived sample used by both core and combat. These checks do not certify unavailable synchronous objective/health features.

The run binds datasets, resource/feature caches, chronological baseline hashes, source snapshots, models and predictions. A supplemental dependency audit verifies imported residual helpers and feature modules against hash-bound source snapshots completed before fitting. No outcomes after September 2 were opened.

[Full machine-readable report](../data/wpx/combinations_2026-10-04/report.json), [selection](../data/wpx/combinations_2026-10-04/selection.json), [direct comparison with SQ](../data/wpx/combinations_2026-10-04/versus_best_single.json), [completion hashes](../data/wpx/combinations_2026-10-04/completion.json), [runner](../scripts/wpx_combinations.py).

Reproduction uses the completed source snapshot and hash-pinned local caches. A fresh run is `python3 scripts/wpx_combinations.py --out data/wpx/NEW_OUTPUT_DIRECTORY`. Existing outputs are preserved; `--resume` accepts only the identical plan and validated completed candidate artifacts.

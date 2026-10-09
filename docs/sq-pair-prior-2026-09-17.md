# Solo-queue matchup/synergy prior as one input to our draft model — 2026-09-17

**Result: the signal is real and clears the house bar.** One solo-queue-derived pair score, added to our walk-forward draft model with a walk-forward slope, improves log loss by **−0.0023 (95% CI −0.0043 to −0.0004)** on 4,122 pro maps using only solo-queue data from *earlier patches*. Fitted slope on our native model 0.60 [0.36, 0.86], z = 4.6. For scale: our entire draft model is worth −0.0049 over controls on the same maps, so this is ~+47% on top of it from one parameter.

Follows [the TGV head-to-head](tgv-head-to-head-2026-09-17.md), which left TGV's pair-interaction edge unresolved for lack of clean maps. Solo-queue stats contain no pro outcomes, so every covered pro map is a clean test.

## Data and score

- Source: Lolalytics public JSON (emerald+, all regions, ~25–29M games/patch), patches 16.8–16.17, the 291 champion/role combos with ≥3 pro picks; 12,356 of 12,360 pages fetched at ~1 req/s, no blocks (`scripts/sq_pairs_scrape.py`, `data/sq/lolalytics/`). robots.txt permits; u.gg was bot-walled and left alone.
- Effect per pair = 0.04 × Lolalytics `d2` (verified: pair win rate in pp beyond both champions' baselines, `vsWr − (100 − wr_b) − (wr_a − 50) − (avgWr − 50)`; 0.04 converts pp → logit). All 25 lane-vs-lane matchups and 10 same-team synergies per side.
- Shrinkage n/(n+k), k = 4/τ² with τ² by method of moments on solo-queue data alone: τ ≈ 0.050 (matchups, k ≈ 1,570 games), τ ≈ 0.029 (synergies, k ≈ 4,800). No pro data touches the score.
- Draft score = Σ matchups + Σ blue synergies − Σ red synergies. 99.4% pair coverage; sd 0.26 logit; correlation with our own `syn:`/`vs:` terms 0.13.

## Results (`scripts/sq_pairs_eval.py`, `data/sq/eval/report.json`)

Ours and controls refit walk-forward at half-month cutoffs from 2026-05-01; cluster bootstrap over team-pair/date.

| Patch convention | Maps | Slope on our model | z | Walk-forward slope Δ log loss vs ours | Fixed half-slope |
|---|---:|---:|---:|---:|---:|
| **prior patches only (no look-ahead)** | 4,772 | 0.60 [0.36, 0.86] | 4.6 | **−0.00227 [−0.00425, −0.00036]** | −0.00241 [−0.00403, −0.00077] |
| same patch only | 4,866 | 0.60 [0.31, 0.88] | 4.3 | −0.00157 [−0.00324, +0.00018] | −0.00168 [−0.00307, −0.00028] |
| pooled through same patch | 4,866 | 0.60 [0.37, 0.83] | 5.0 | −0.00269 [−0.00472, −0.00064] | −0.00256 [−0.00434, −0.00088] |

- Pooling patches beats a single patch (pair effects are kit-driven and stable; more n wins), and the strictly-prior pool loses almost nothing — so there is no live look-ahead problem.
- The walk-forward slope is stable at 0.5–0.65 in every fold after the first: the raw score is ~1.7× too large (MoM shrinkage is not enough; solo-queue effects transfer to pro at ~60%).
- It is the **matchups** (z = 5.1); synergies point the same way but are individually weak (z = 1.7).
- Minor leagues carry it (slope 0.69, z = 4.8); major leagues are positive but unproven (0.37, z = 1.5, n = 1,320) — same pattern TGV showed. Sept 4+ block: slope 1.04, z = 3.4.
- At slope 1 (unscaled) the gain is not significant (−0.0014 [−0.0047, +0.0019]) — the "heavily shrunk" part of the hypothesis matters.

## Reading

TGV's pair-table edge was probably real and this recovers the clean part of it with public data. It does not make TGV's model better than ours overall — it identifies the one ingredient we lacked.

## Follow-up (same day): shipped into the draft outcome model; live-GAM check; nightly refresh

**1. `draft.fit_outcome_model` — SHIPPED (`draft.SQ_PAIR_ENABLED = True`).** Production code is `lol_ticker/sqpairs.py` (scrape/refresh, compact `data/sq/pair_tables.npz`, `Scorer` using prior patches only; agrees with the research script to 1.4e-4). The score enters as one extra column `__sq_pair__` = score / 0.25 (≈ its sd, so λ=300 means the same as for the 0/1 indicators), ridge-penalised with everything else, stored in `draft_outcome_model` and the fit meta; unavailable score = 0. The draft edge fed to wpx (`draft_outcome_oos`) now includes it. New kwargs `holdout_since` / `write=False` make the production trainer gateable.

Newest-date gate (`scripts/sq_pair_gate.py`, production trainer, cluster bootstrap):

| Train before | Holdout maps | Learned slope on raw score | Log loss without → with | Paired Δ log loss | Paired Δ Brier |
|---|---:|---:|---|---:|---:|
| 2026-08-01 | 1,972 | 0.43 | 0.63310 → 0.63034 | **−0.00276 [−0.00496, −0.00054]** | −0.00135 [−0.00236, −0.00034] |
| 2026-07-16 | 2,580 | 0.34 | 0.63237 → 0.62969 | **−0.00268 [−0.00429, −0.00104]** | −0.00125 [−0.00197, −0.00053] |

Both intervals exclude zero. For scale, the whole draft vocabulary is worth −0.0027 over Elo-only on the first holdout, so this one input doubles the draft model's edge. The slope is learned from only ~2,000–2,900 covered training maps and will firm up (toward ~0.6) as coverage accumulates.

**2. Live GAM pregame prior — survives, NOT absorbed, but under the bar to roll the contract** (`scripts/sq_pair_live_check.py`: production `wpgam.fit_arrays` with temporal calibration on 14,390 gol.gg games before 2026-08-01; 1,145 covered newer games, major regions).

- On top of the GAM pregame prior (team + champion logits): slope learned forward on out-of-fold training logits = 0.56; in-block 0.50 [0.09, 0.97], z = 2.1. Correlation with `prior_champ_logit` only 0.17 — the GAM's champion terms are not already carrying it. Forward gain −0.0020 log loss, CI [−0.0060, +0.0023]: right size, not significant on 1,145 major-region games (the slice where the signal was always weaker).
- On top of the full in-game GAM logit the signal decays with game time and is gone by ~15–20 min:

| Minute | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| slope | 0.56 | 0.59 | 0.47 | 0.30 | 0.15 | −0.15 | 0.04 |
| z | 2.4 | 2.3 | 1.6 | 0.9 | 0.4 | −0.3 | 0.1 |

Reading: draft matchup advantage is real pregame information that the gold/objective state absorbs as it is realised — exactly how `prior_champ_logit` behaves. If it goes into the GAM it should be a fourth monotone `PRIOR_INPUTS` channel with its own time-knot curve, not a constant offset (a constant slope over all states is z = 1.2 and hurts after 20 min). Not shipped: needs a contract/ledger roll and the house bar is a CI excluding zero; re-run this script when ~2.5k covered gol.gg games exist (≈ end of Worlds).

**3. Nightly refresh — DONE.** `python3 -m lol_ticker sqpairs refresh` finds the newest pro patch, probes it and the next two on Lolalytics, re-fetches pages of a still-live patch at most every 6 days (~1,240 requests, ~25–30 min at 1 req/s; most nights zero), writes a `.final` marker once a newer patch exists, and rebuilds the tables; it aborts on 403/429. Runs as soft-failing phase S in `scripts/update.sh`, before `draftfree`. First run fetched live patch 16.18. Season rollover (17.1) is not probed automatically.

Tests: `tests/test_sqpairs.py` (prior-patch-only pooling, side-swap antisymmetry, zero on unknown inputs, key normalisation); full suite 244 passing.

## Remaining
- Re-gate the GAM channel at ~2.5k covered gol.gg games (script above).
- Major-league-specific slope once n allows.
- Coverage starts at patch 16.8 (Lolalytics keeps ~10 patches) — the nightly refresh is what preserves history from here on; do not delete `data/sq/lolalytics/`.

```sh
python3 -m lol_ticker sqpairs refresh
PYTHONPATH=. python3 scripts/sq_pair_gate.py
OPENBLAS_NUM_THREADS=4 PYTHONPATH=. python3 scripts/sq_pair_live_check.py   # ~10 min
```

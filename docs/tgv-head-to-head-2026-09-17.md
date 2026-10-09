# TGV vs our champion model: fidelity check, then head-to-head — 2026-09-17

**Verdict: unresolved, with TGV's pair terms the more promising unknown.** On the only maps neither side could have trained on (476 maps, Sept 4–15) the two are statistically indistinguishable. TGV's large historical win is mostly training-set fit, demonstrated below — but a residual out-of-sample signal survives and ours shows none on that block.

## 1. Is the TGV model recreated faithfully?

Partially: **exact where exported, ~86% of the score on real pro drafts.**

| Check | Result |
|---|---|
| Unit tests (`tests/test_tgv_reconstruction.py`) | 4/4 pass |
| Daily terminal draft: strength, comfort, matchup, synergy | error 0.0 each; total matches review to 0, engine to 1.1e-8 (side + composition supplied, not recovered) |
| 56 published checksum drafts, unary+matchup+synergy | corr 0.997, residual sd 0.091 logit (inflated by artificial drafts with unary sd 1.2) |
| **New: 281 teams' published last-10 median draft score, rebuilt from our OE drafts** | **r = 0.933, R² = 0.86, RMSE 0.058 vs signal sd 0.157** |

The last row is the first end-to-end test on real professional drafts (`research/tgv_fidelity_gaps.py`; team IDs are OE IDs, game counts align for 281 of 305 matched teams). Build-up: unary alone r = 0.42 → + matchup + synergy r = 0.895 → + recovered comfort curve with historical counts r = 0.933 (export-time counts 0.926, so historical timing fits slightly better). The remaining ~14% of variance is the unrecovered composition term plus game-set mismatches. Everything below therefore tests **TGV minus composition**; side bias is irrelevant because our controls carry an intercept.

Notable: in pro drafts most of TGV's draft variance is **matchups/synergy, not champion strength** (score sd 0.35 vs unary 0.17).

## 2. Head-to-head on historical maps (biased toward TGV)

`research/tgv_vs_ours_headtohead.py`: 2,885 maps, 2026-06-17 → 09-15, patches 16.12–16.17. Ours = outcome draft model refit walk-forward at half-month cutoffs (every map out of sample). TGV = fixed Sept 15 release (has seen every map before Sept 14). Both scores are added, unscaled, to the same controls-only offset (team Elo + player history). Log loss change vs controls, cluster-bootstrap 95% CI:

| Score | Δ log loss | Fitted slope (1 = well scaled) |
|---|---:|---:|
| ours: picks only | −0.0026 [−0.0052, −0.0000] | 1.07 |
| ours: picks + pairs | −0.0035 [−0.0068, −0.0003] | 0.97 |
| ours: + bans (native) | −0.0046 [−0.0079, −0.0012] | 1.05 |
| TGV: unary only | −0.0015 [−0.0042, +0.0014] | 0.75 |
| TGV: unary + matchup + synergy | −0.0095 [−0.0150, −0.0039] | 0.88 |
| TGV: + comfort curve | −0.0105 [−0.0160, −0.0049] | 0.94 |
| 50/50 blend | −0.0100 [−0.0135, −0.0066] | — |

TGV − ours (champion terms): Brier −0.0030 [−0.0058, −0.0003]. Joint regression keeps both (ours 0.74 [0.32, 1.16], TGV 0.82 [0.58, 1.06]); the scores correlate only 0.16 (unaries −0.01), so they carry different information. Unary vs unary is a tie everywhere — **TGV's whole apparent edge is its pair tables.** The edge is absent in major leagues (773 maps: +0.0005 vs ours −0.0025, both CIs span zero) and concentrated in minor leagues and in September.

## 3. Leak test: that edge is mostly in-sample fit

`research/tgv_leak_test.py` scores the same maps with TGV's **older Sept 3 release** (all 865 relation files were already scraped). Slopes refit per block so the probit/logit difference cancels.

| Block | Legacy release (Sept 3) | Current release (Sept 15) | Ours (always OOS) |
|---|---:|---:|---:|
| Before Sept 3 — both TGV releases in-sample (2,367 maps) | **−0.0179** (z 8.4); unary alone −0.0114 (z 7.0) | −0.0084 (z 6.3) | −0.0042 (z 4.2) |
| After Sept 4 — legacy out-of-sample (476 maps) | **−0.0051** (z 2.1); unary alone −0.0004 (z 0.6) | −0.0179 (z 3.7, in-sample) | −0.0010 (z 1.0) |

The legacy model loses ~70% of its gain and all of its unary signal the moment it leaves its training window, while the current release posts its best numbers exactly on the maps closest to its own cutoff. The two releases correlate only 0.37 on pro drafts (0.12 on raw matchup cells), so the tables are being substantially refit to pro outcomes release to release — this is not a frozen solo-queue prior with six transfer scalars. The §2 table is therefore not evidence that TGV predicts better.

## 4. The clean comparison

Legacy release as published (probit × scoreScale × 1.702), 476 maps / 147 clusters, Sept 4–15, never seen by either model:

| | Log loss | Δ vs controls |
|---|---:|---:|
| Controls | 0.63212 | — |
| Ours | 0.63196 | −0.0002 [−0.0074, +0.0075] |
| TGV legacy | 0.62947 | −0.0026 [−0.0195, +0.0132] |
| 50/50 blend | 0.62889 | −0.0032 [−0.0141, +0.0075] |

TGV − ours: −0.0025 [−0.0133, +0.0093]. Point estimate favors TGV; the interval is ~5× wider than the difference. The 40 maps after the current release's as-of date say the same nothing (−0.004 [−0.047, +0.047]).

## What to take from this

- Neither champion model demonstrably beats the other out of sample. Ours is the better-scaled and honest-by-construction one; TGV's is higher-variance (2× our score sd) with a possibly real pair-interaction signal (OOS slope z = 2.1) that ours lacks — our `syn:`/`vs:` indicators need ≥60 pro co-occurrences, theirs come from 6.9M solo-queue games.
- Unary champion strengths: a tie, and uncorrelated. No reason to import theirs.
- The actionable idea is a **solo-queue-derived matchup/synergy prior** entering our model as one shrunk score, not their coefficients.
- Resolving the question needs ~2–3k clean maps. The Sept 15 release is frozen on disk: re-run `tgv_vs_ours_headtohead.py` with `tgv_out_of_sample` as the only slice once Worlds/autumn maps accumulate, and do not re-scrape a newer release into the same comparison.

Caveats: TGV scored without composition (§1); controls are the compact Elo/player set, not the deployed priors; slopes in §3 are fit in-block (one parameter); September blocks are all patch 16.17. Database read-only; no model changed.

```sh
PYTHONPATH=. python3 research/tgv_fidelity_gaps.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 research/tgv_vs_ours_headtohead.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 research/tgv_leak_test.py
```

Outputs: `data/tgv/20260916/fidelity-draft-gaps.json`, `data/tgv/head-to-head/{report,predictions,leak-test}.json`.

# Isolated draft plus player comfort versus TGV

Follow-up: [reverse engineering the public model assets](tgv-reverse-engineering-2026-09-16.md) recovered TGV's saturating count-based comfort formula. The proxy experiment below does not implement that formula and should not be interpreted as a test of TGV's actual comfort estimator.

The closest comparable output is a neutral-team draft advantage: champion selection plus player–champion comfort, with team strength and blue-side bias removed. We now have a reproducible isolated scorer and 4,358 historical per-map decompositions. The comfort proxies add no convincing predictive improvement after controlling for overall player strength and experience. TGV's public material still does not provide matched forecasts or validation sufficient for a numerical accuracy comparison.

This corrects the earlier comparison: `scripts/wpx_playerchamp.py` already experimented with player/champion win history, experience, unfamiliar picks and GD@15. Those experiments were not part of the deployed draft model. Calling comfort an entirely new idea for this repository was too broad.

**What was isolated**

The new offline diagnostic uses the outcome-based draft vocabulary in `lol_ticker/draft.py`, not the market-repricing regression or live-state GAM. Its contributions are:

| Component | Included information |
|---|---|
| Draft | Champion picks and bans, role and patch deviations, same-team synergies and cross-team matchups |
| Comfort | Earlier player–champion–role win rate relative to the player's overall role win rate; log prior champion appearances; never-before-played champion count |
| Nuisance controls, removed from isolated output | Team Elo, players' overall win history, total experience, rookie count, and intercept/side bias |

Full match probability is sigmoid(intercept + controls + draft logit + comfort logit). The isolated output is **sigmoid(draft logit + comfort logit)**, and its neutral advantage is that probability minus 50%. This is a model attribution conditional on the fitted controls, not a causal estimate of what switching a pick would do.

To make the isolated quantity side-neutral, each existing draft feature is represented as half its blue-perspective value minus half its red-perspective value. Reversing teams therefore negates every draft and comfort contribution. This is an offline refit using our existing feature vocabulary with explicit symmetry, not extraction of the exact deployed coefficients. The existing stored draft model remains unchanged.

Comfort uses the previous experiment's 5-game shrinkage toward a player's overall rate, itself shrunk with 10 games toward 50%. Keys now include role. All history is lagged by a complete UTC day, so neither current-game nor same-day outcomes can enter features. Current-game GD@15, live telemetry, market prices and game results are excluded from prediction inputs. GD@15-based historical comfort was not tested in this screen. The measured proxies should not be equated with intrinsic champion mastery.

**Chronological results**

The read-only database snapshot yielded 28,463 complete maps before September 3, 2026; 47 records were rejected for incomplete or inconsistent rosters/drafts. Fit used 20,332 maps before January 16, 2026; validation used 3,773 maps through April 30; the later diagnostic used 4,358 maps from May 1 through September 2. Coefficients stay frozen after the training cutoff, while historical features update using earlier dates. Fixed ridge penalty is 300; support selection and scaling use training data only. No tuning or probability calibration was performed.

All models below receive identical team/player controls. Metrics evaluate full outcome probabilities, not the neutralized attribution, because real opponents have unequal strength. Lower is better.

| Model | Validation Brier | Later Brier | Later log loss |
|---|---:|---:|---:|
| Team/player controls | 0.209155 | 0.212630 | 0.613494 |
| Controls + draft | **0.207657** | 0.211791 | 0.611429 |
| Controls + draft + comfort | 0.207804 | **0.211592** | **0.611081** |

Comfort minus draft-only Brier:

- Validation: **+0.000147**, paired 95% interval **[−0.000385, +0.000659]**.
- Later diagnostic: **−0.000199**, paired 95% interval **[−0.000675, +0.000305]**.

Intervals use 2,000 paired bootstrap draws over team-pair/UTC-date clusters (1,952 validation and 1,918 later clusters). Both include zero. The earlier validation block favors draft without comfort. These are exploratory historical results, not a fresh prospective promotion test or evidence of superiority over TGV.

The initial run with only team Elo as a control produced an apparent −0.003720 later Brier gain from comfort. After adding overall player controls, the gain fell to −0.000199. That large reduction shows why player experience and strength must be accounted for before interpreting champion familiarity. The parent artifact directory preserves that initial run, but the `player_controlled` results are the primary comparison.

In the later block, adding comfort moves the isolated probability by a median absolute **1.35 percentage points**, holding the joint model's draft coefficients fixed. Draft alone in that joint model has a median absolute neutral advantage of **2.86 points**. Only **1,592/4,358 maps (36.5%)** have prior champion appearances for all ten players. The win-history term is positive; the log-experience term is nearly zero after player controls; an additional unseen champion contributes approximately −0.028 log odds, conditional on other terms. These magnitudes are not accuracy measures.

**Comparison with TGV on the same scope**

| Draft-specific capability | Our isolated scorer | TGV public model |
|---|---|---|
| Role/patch champion strength | Explicit regularized effects; unseen patches fall back to base/role effects | Role-specific strength and patch history |
| Synergy and matchups | Supported pair indicators, estimated from outcomes | Exposed relationship scores |
| Player–champion comfort | Defined historical proxies with shrinkage and earlier-day provenance | Actual comfort in professional rankings; generated comfort in puzzles; estimator undisclosed |
| Explicit composition | No separate tankiness or damage-mix representation | Composition terms include tankiness, damage output and balance |
| Neutral draft advantage | Team/player controls and side intercept removed | Rankings describe a neutral 50% baseline, side bias removed and actual comfort included |
| Search | Completed-draft scorer | Claims minimax search within constrained pools |
| Measured prediction quality | Chronological metrics and paired intervals above | No comparable numerical benchmark found |

Sources: [TGV Champion Board](https://tgv-analytics.com/champions), [Daily Draft explanation](https://tgv-analytics.com/arcade), and [rendered Rankings inspected in the preceding scrape](https://tgv-analytics.com/teams). Their training population, comfort estimator, coefficient regularization and probability calibration remain unknown.

TGV's current ranking examples—Gen.G +0.7 points, G2 +4.6, KT −6.9—aggregate the median of recent draft advantages. They cannot be directly compared with our per-map effects or our May–September evaluation: the underlying games, model versions and training windows differ. No rank correlation or head-to-head accuracy claim is made. A proper comparison requires both scorers' probabilities on identical completed drafts, rosters, patches and forecast dates. Today's reconstructed rankings are insufficient for a historical out-of-sample comparison.

The meaningful remaining structural gaps are explicit composition features and TGV's decision search. Our comfort experiment already covers simple familiarity; this run gives no evidence to promote those proxies. TGV may estimate comfort better, but its public interface cannot establish that.

**Artifacts and verification**

- Runner: `scripts/draft_comfort_compare.py`.
- Primary results: `data/wpx/draft_comfort_comparison_20260916/player_controlled/report.json`.
- Saved coefficients/scales: `player_controlled/model.npz` beneath the same directory.
- Per-map draft/comfort logits, neutral probability and full probability: `player_controlled/scores.json`.
- Source snapshot and manifest: `player_controlled/source.py`, `player_controlled/manifest.json`; input snapshot: parent `inputs.json`.

Three tests passed for same-day/future outcome isolation, side reversal, and unseen-player handling. All 4,358 saved neutral scores reconstruct from their component logits and reverse to complementary probabilities within 1e-14. The three final fits converged and outputs are finite. Production models, database contents and forward evaluation registries were not modified.

Reproduce into a new output directory, without database access:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/draft_comfort_compare.py \
  --input data/wpx/draft_comfort_comparison_20260916/inputs.json \
  --out data/wpx/draft_comfort_reproduction
```

Limits: the control model is intentionally compact and does not reproduce all deployed team/player priors; fixed shrinkage is not an exhaustive comfort search; player/team identity changes, missing history and roster confounding remain; patch terms cannot learn unseen patch effects; the split is chronological but uses already available historical outcomes. These findings apply to this defined scorer and population.

# Corrected inputs and a frozen model challenger

The September 11 research has been implemented as an isolated corrected dataset, a narrow resource-model experiment, and a forward candidate evaluator. The deployed v8 model remains fixed. Historical correctness repairs and evidence of better future predictions are separate claims.

## Historical input repair

The final dataset is `data/wpx/states_inputs_v2_canonical_before_2026-09-03.npz`, SHA-256 `c539d0180028a13137d1995e97d572dfbfe551d88699c67517c60bd5e9ced63d`. It contains the same 15,408 consumed games, 500,122 fixed states and 1,776,740 total states as the legacy dataset restricted through September 2. Game IDs, outcomes, timestamps and champion slots match that population exactly. Newer outcomes present in the default dataset were excluded from development.

All 2,609 original feed openings were recovered with cached provenance. An opening can be pregame, so its wall timestamp is not treated as the exact game clock. The builder brackets observations using all ten players' cumulative gold, verifies opening champion/role identity with canonical champion aliases, and combines that interval with the original wall-time bound. It joins HP only when the entire admissible observation interval precedes the prediction and the worst-case age is at most 90 seconds. Ambiguous clocks and detected resets are unavailable. Event-anchored pre-event rows use the actual earlier prediction second.

An independent source audit checked 117,468 retained HP states across 2,347 games and 45,898 distinct observation intervals: **zero future joins, zero raw-feature mismatches and zero gold-bound violations**. Fixed-state HP coverage falls from 82,318 (16.46%) to 20,511 (4.10%). This is a conservative reduction in usable historical telemetry, not a claim that the feed became less complete. Live current-frame HP remains available.

The sparse legacy arrays lack participant IDs for every observation. Stable slot ordering and the absence of an entirely missed remake with the same champions remain explicit assumptions; the repair does not certify exact pause/attempt reconstruction. New imports preserve identity and counter metadata. See the dataset manifest and `data/wpx/inputs_v2_canonical_source_audit_2026-09-12.json`.

Estimated deaths now track unique victims rather than count recent kills; the 55 fixed states with more than five dead players on one side become zero. The dataset records death source and uncertainty. Inventory handling recognizes sale/destruction and patch-specific prices without a current-patch fallback, but stored undo events lack the item identities required for exact reconstruction. The corrected modeling contract therefore explicitly zeros item-completion and item-gold channels. These channels affect champion-state nuisance fitting, not the main GAM's direct features.

## Model comparison

The study compares repaired core, recorded v8 state architecture, the existing rich control, shared role resource curves, constant champion deviations and smooth champion deviations. Blue and red share resource functions. Champion deviations can be negative while the complete physical gold response remains nonnegative, including at clipping boundaries. Transforms and champion support are fitted only on training games.

Three development blocks select two shrinkage settings and none/temperature/Platt calibration. Each calibration block is the preceding 28 days; the complete pregame and champion-state stack is rebuilt using earlier games. Selection favors the core unless a pooled family has a negative paired series-bootstrap upper bound and improves every leave-one-development-block average. This rule is development selection, not confirmatory evidence. The later frozen and monthly replays retain those choices.

**Development selected the repaired core with Platt calibration**, mean block Brier 0.148544. None of the pooled families passed the robustness rule. The later replay retains this selection.

Both later comparisons cover 3,097 games from May 1 through September 2. Lower game-balanced Brier is better. Intervals below are paired whole date/match bootstraps on monthly replay; they are exploratory because these outcomes were already consumed.

| Model | Frozen Brier | Monthly Brier | Monthly minus core [95% interval] |
|---|---:|---:|---|
| Repaired core | 0.144172 | 0.143550 | +0.000000 [+0.000000, +0.000000] |
| Recorded v8 state architecture | 0.144239 | 0.143616 | +0.000066 [+0.000016, +0.000117] |
| Rich control | 0.144797 | 0.142943 | -0.000606 [-0.001516, +0.000299] |
| Shared role curves | 0.144168 | 0.143543 | -0.000006 [-0.000026, +0.000013] |
| Constant champion deviations | 0.143669 | 0.143219 | -0.000331 [-0.001101, +0.000409] |
| Smooth champion deviations | 0.143546 | 0.142962 | -0.000588 [-0.001491, +0.000331] |

The repaired core has a very small advantage over the v8 architectural control. The rich and champion-resource point estimates improve in monthly replay, but their intervals include zero and they did not earn selection on earlier blocks. These scores do not establish that the final frozen candidate beats the currently deployed v8 artifact.

The v8 reference uses the recorded v8 state architecture with the same corrected inputs and chronological upstream/calibration blocks as the other models. It is an architectural control, not a reproduction of every historical deployed-v8 training decision.

Reproduction:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 research/wpx_resource.py \
  --dataset data/wpx/states_inputs_v2_canonical_before_2026-09-03.npz \
  --output data/wpx/resource_study_reproduction_20260912 \
  --resources data/wpx/action_20260911/resources \
  --v8-source data/wpx/repairs_v9/incumbent_source.py
```

The completed results are in `data/wpx/action_20260911/resource_study`. The reproduction command uses a new directory because the recorded study source predates the native serving-calibration extension.

`plan.json` binds dataset, input manifest, resource rows, source snapshots and protected production artifacts. Stage caches bind fitting game IDs, the plan and their actual contents. Predictions and fitted models have hashes. Resuming a changed plan requires a new output directory. Every frozen and monthly model receives coherent individual-player gold perturbations; diagnostic controls may fail, but a new candidate must pass.

## Forward evaluation and operation

The candidate ledger records one frozen artifact on the exact newly captured incumbent frames, independent of exchange-quote availability. It never backfills predictions onto old frames. Registrations and forecasts are immutable; artifact, input and inference-source hashes are checked. Collection rejects games that started before registration, known outcomes, changed source/artifact generations and predictions outside the fixed deadline. A bounded asynchronous worker isolates candidate inference from incumbent capture.

The fixed evaluation rule compares game-balanced Brier with whole team-pair/date clusters, separate incumbent artifact/source cohorts, log loss, calibration error, a minimum useful gain and a precision requirement. It makes one promotion decision after the declared endpoint. Interim scores are descriptive and do not trigger promotion. The 100-game floor alone is not a power calculation.

The shared outcome inventory imports both legacy registries without changing them. It initially marks all 15,408 consumed games and subsequently adopts already-inspected live outcomes, including frames without valid market quotes. Unknown feed-to-gol.gg links conservatively quarantine their dates. Changing an evaluation registry cannot make these outcomes fresh. Every model refresh stages; promotion requires valid exact-artifact forward evidence and copies that artifact without a final refit.

Live input fixes use the actual OE rating-source roster for player priors, add availability/age/resolved-player provenance and absolute gold for all ten players, bound feed fetching, and preserve state across repeated or overlapping windows. Numerical prior shrinkage was not changed without an ablation. The nightly update retains source ingestion, ratings and shadow scoring, while automatic GAM building/fitting/historical scoring is gated during this transition. Explicit historical builds require a new output path and cutoff; `wpx all` cannot mix corrected and legacy contracts.

## Final artifact and verification

The exported core is `data/wpx/action_20260911/final_candidate/candidate.npz`, SHA-256 `e8e61aace6a236ea5a15e63ef7bacc31d8be1e41998b2163c08df4d54d3f3849`. It fits 14,585 games through August 5 and calibrates on 823 separate games from August 6 through September 2. No state or nuisance refit follows calibration. Its held-out Platt intercept is 0.291642049 and slope 0.944545996.

Native optional probability clipping preserves the study's exact calibration semantics in both array and production inference. Older artifacts have no clipping field and retain bit-identical outputs. This serving extension was added after the historical audit. The prior-cache transition records old/new source hashes and verifies that all upstream code, constants and imports are unchanged; fitting sources and serving sources are archived separately. The final adapter manifest additionally binds `util.py`, including team normalization. Unsupported experimental adapters cannot be promoted.

Validation passed:

- 216 tests in the full suite, including real PostgreSQL candidate-ledger tests; the final dependency-manifest change also passed all 12 candidate tests.
- 57 historical fitted artifacts and their saved predictions/metrics verified; 84 independent convex calibration refits changed Brier by at most `5.48e-9`.
- 171,660 coherent gold comparisons per family across frozen/monthly artifacts: zero reversals for repaired core, rich and all pooled families; the v8 control has 182 reversals.
- 329,976 Elder/death comparisons across all 30 frozen/monthly models: zero reversals. Missing-HP fallbacks are finite and consistent with zero telemetry.
- Final candidate: exact serialization roundtrip, zero reversals in 60,000 additional gold comparisons, historical/live probability difference below `6.2e-9`.
- Independent serving check: existing v8 probabilities/components are bit identical on 1,000 raw cases; 687 live cases verify both old-artifact compatibility and exact candidate/production parity. The standard production shape audit passes.

## Registered forward experiment

Candidate `candidate-a5b5861d33449d87a3773646ad0cc306843436b3d50c5de5c3732388ea2080e7` was registered at **September 12, 06:47:25 UTC** (September 11, 23:47 PDT). Its read-only copy is in `data/wpx/candidates/`; the full registration, input contract, inference snapshots and sample-size basis are saved in `data/wpx/action_20260911/`.

| Frozen requirement | Value |
|---|---|
| Endpoint | December 1, 2026, 00:00 UTC |
| Minimum mutually valid games per incumbent cohort | 2,437 |
| Maximum 95% interval halfwidth | 0.002 Brier |
| Minimum worthwhile gain | Upper interval bound below −0.001 candidate minus incumbent |
| Supporting checks | No positive game-balanced log-loss or ten-bin calibration-error regression |
| Population | Games starting after registration; identical captured model frames; no market quote required |

The exact-artifact planning pilot uses each model's own upstream priors on consumed August 6–September 2 rows. Its clustered standard error is 0.001242, implying approximately 1,219 games for the desired halfwidth at observed variance, or 2,437 with doubled variance. This is much larger than the misleadingly small variance of architectural controls sharing one upstream stack. Calibration reuse and overlap with the incumbent's fitting outcomes prevent treating the pilot as held-out accuracy evidence. Its point delta is +0.002071 Brier; it was used for planning variance, not to select or promote a model. The prospective population can differ, so the actual precision requirement remains binding. A shortfall at the endpoint is inconclusive; there is no automatic extension or promotion.

The old shadow recorder PID 8018 was terminated and replaced by PID 26909. The new process loaded exactly one frozen registration, completed maintenance, wrote the candidate report and captured a real live incumbent frame. An inference smoke check on that live state finished in 0.003 seconds; it is explicitly excluded from prospective evidence and inserted no forecast row. At verification there were **zero eligible candidate forecast rows** because the current game started before registration. The collector is ready for later games; no prospective accuracy claim is available yet.

Inspect with `python3 -m lol_ticker.wpcandidate status`. Normal recorder maintenance and nightly shadow scoring refresh `data/wpx/shadow_candidate_score.json`. Changing frozen inference source or the artifact stops candidate scoring rather than silently changing the experiment.

The original `states.npz`, both deployed model files, `live_stack.json` and both legacy evaluation registries retain their starting hashes. The new shared exposure inventory and immutable candidate tables are intentional additions. Historical experiments, final training and source snapshots are isolated; the deployed v8 artifact remains unchanged.

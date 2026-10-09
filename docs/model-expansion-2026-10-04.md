# Model expansion: ideas 2–7, October 4, 2026

Six research implementations were developed in parallel by three workers, with
shared capture and experiment infrastructure integrated separately. These are
staged candidates and richer input capture. Production model weights and
evaluation registrations are unchanged.

| Idea | Implementation | Current evidence and coverage |
|---|---|---|
| 2. Objective opportunities | `wpobjective`: remaining Baron/Elder time, living-player/resource coupling, causal acquisition intervals, optional spawn/soul windows | Remaining-buff candidate passed development selection; frozen ΔBrier −0.00000185, interval [−0.00000614, +0.00000267]. Historical spawn/readiness couplings remain unavailable. |
| 3. Composition traits | `wpcomposition`: four literal Data Dragon proxies plus eight reviewed capability features; per-release catalog fetcher and isolated draft candidate | The static-proxy screen retained the frozen draft baseline. Rich engage/disengage/waveclear/damage/scaling annotations have zero historical coverage. |
| 4. Fearless draft pressure | `wpfearless`: certified prior series picks, tournament/phase rules, supported player pools and current-pick support | Implemented and tested; existing draft archives lack certified series/rule relationships, so retrospective accuracy is unevaluated. |
| 5. Prior confidence | `wpconfidence`: source age, actual recent sample support, exact rating-source roster continuity and resolved-player support | An isolated fixed shrinkage policy is ready for captured metadata. The corrected archive cannot reconstruct the required per-side metadata; no historical accuracy claim. |
| 6. Recent trajectory | `wptrend`: bounded causal 30/120/300-second role gold/health/alive and event changes, with volatility summaries | Candidate passed development selection; frozen ΔBrier −0.00006625, interval [−0.00014854, +0.00002000]. Thirty-second and synchronous health histories remain unavailable historically. |
| 7. Early live SQ prior | `wpsqearly`: own nonnegative, nonincreasing time curve ending exactly at minute 20 | August paired Brier change −0.0004243, interval [−0.0012158, +0.0004331]; baseline retained. September's 19 maps are a diagnostic. |

## Capture integration

`wpresearch` attaches the following blocks to the live state saved by the
shadow recorder: `objective_opportunities`, `composition`, `fearless`,
`prior_confidence`, `trajectory`, `sq_early`, and `draft_snapshot`. They do not
enter either production feature vector or apply any candidate probability.
The existing role-specific `combat` capture remains in place.

`draft_snapshot` preserves official team IDs, participant IDs, stable esports
player IDs when supplied, champions, roles, summoner names, series ID and game
number. This is raw source capture, not a certified identity bridge or a
claim that the tournament uses Fearless. The Fearless adapter verifies the
actual current metadata against a certified bundle; a bundle for another
known series or game is rejected.

Composition capture loads local exact-release catalogs under
`data/composition/catalogs/VERSION.json`, with optional reviewed annotations
under `data/composition/annotations/VERSION.json`. The fetcher saves raw primary
source responses and content hashes, using conservative HTTP Last-Modified
availability times. Mode-specific champion variants are excluded. Capture
never downloads a mutable latest-patch catalog.

A full server-build version is retained in `draft_snapshot` and is not
silently treated as an exact Data Dragon release. An explicit
`research_context.composition_patch` can select a verified two-component patch
or release; the block records both the original feed patch and the mapping.
Riot documents that client and Data Dragon versions can differ, and that a
patch may have multiple Data Dragon builds. [Riot Data Dragon versions](https://developer.riotgames.com/docs/lol#data-dragon).

The SQ scorer normalizes official release versions such as `16.16.1` to the
gameplay patch `16.16` before pooling tables, so same-patch statistics cannot
be included merely because the feed supplies a third version component.
Missing coverage, a valid zero score, and the post-20-minute baseline fallback
remain distinct conditions.

Prior confidence receives the full side ratings/provenance before inference
filters numeric priors. Recent counts describe completed rows selected by the
existing 120-day source query; they are not lifetime counts or necessarily a
full 120 days preceding a stale rating. Exact adjusted player means are
preserved. The confidence as-of is the wall time when the prediction inputs
are obtained, not the delayed feed frame timestamp. Repeated identical prior
inputs reuse the same confidence block. A remake resolves the new lineup
before calculating priors and confidence.

Trajectory collection uses the existing pause-aware game clock. The recorder's
15-second laps fetch 10-second windows, so capture allows a 20-second maximum
gap and a causal anchor up to 15 seconds before the requested horizon. Each
window reports `actual_window_s`: a requested 30-second history can therefore
span 30–45 seconds. This is explicitly labeled
`recorder_sparse_windows_v1`. The historical fixed-minute screen uses exact
anchors and a separate 90-second gap bound; that fitted adapter is not silently
served on the approximate recorder windows. Pauses cannot create game-time
history, retries cannot replay events, and remakes/counter regressions reset
the appropriate trackers.

Callers can pass optional `research_context` to `live.estimate_series`, including
`objective_timing_rules`, composition source overrides, or a `fearless` bundle.
Without an explicit bundle, the recorder supplies the known series/game number
and the Fearless adapter can load `data/fearless/series/SERIES_ID.json` with a
dated player-pool snapshot. Missing, mismatched or malformed optional sources
leave their features unavailable while incumbent inference continues.

## Experiments and verification

`wpresidual` fits arbitrary-width, game-balanced time-varying corrections on a
fixed baseline logit. It uses training-only RMS scaling, no centering,
intercept, availability coefficient or baseline refit. Candidate modules bind
their names, input contract and coefficient signs. Artifacts validate their
contract and decay restrictions; a zero correction preserves the caller's
baseline probability exactly, including endpoints.

The historical runners use already-consumed outcomes and existing chronological
baselines. They save candidate weights, predictions, source snapshots, source
coverage and content-hashed completion records into new isolated directories.
Missing-source fallbacks and frozen-baseline reproduction are checked. No
candidate is registered or promoted by these runners.

Read the detailed implementations and screen results:

- [Composition and Fearless](composition-and-fearless-2026-10-04.md).
- [Prior confidence and early SQ](prior-confidence-and-early-sq-2026-10-04.md).
- [Objective opportunities and trajectories](objective-trajectory-research-2026-10-04.md).
- [Previously completed role combat readiness](combat-readiness-2026-10-04.md).

Regression checks cover source/patch identity, future-input exclusion, missing
coverage, side reversal, physical coefficient signs, terminal decay, artifact
roundtrip, recorder cadence, pauses, retries, remakes, malformed optional
sources and known series/game mismatch. Both production feature vectors remain
unchanged by capture. Real-game ingestion depends on a subsequent available
live game; synthetic feed capture and recorder startup are separate checks.

The completed full suite runs 349 tests successfully, with one existing
PostgreSQL candidate-integration class skipped by the sandbox's connection
restriction. `git diff --check` is clean. Objective and trajectory artifacts
each pass 80,000 coherent gold perturbations with zero reversals, exact
zero-extra fallback and side antisymmetry. Their later Brier intervals include
zero, and state-weighted log loss worsens slightly; neither supports promotion.
All five new completion manifests verify their output hashes, and production
weight/stack/registration hashes remain unchanged.

Completed local runs:

- `data/wpx/draft_context_2026-10-04_final_profiles/`
- `data/wpx/confidence_2026-10-04_final/`
- `data/wpx/sqearly_2026-10-04_final/`
- `data/wpx/objective_2026-10-04_screen/`
- `data/wpx/trend_2026-10-04_screen/`

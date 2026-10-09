**Model improvement research — September 11, 2026**

Prioritize historical input correctness and prospective candidate evaluation. The strongest new modeling hypothesis is a constrained GAM that shares resource effects across roles, champions and sides. The evidence does not establish an accuracy improvement from replacing the deployed model with the existing rich GAM, recency model, or a tree/neural model.

Five distinct subagents reviewed evaluation, features, statistical methods, live operation and the final evidence; an additional review challenged the experimental roadmap. The research inspected source, existing experiments, saved live scores, historical input metadata and primary methodological sources. It also independently recomputed the saved scores for all fourteen families and two ensembles. No model was trained or promoted, no database or evaluation registry was changed, and no new outcome-scoring run was initiated.

The historical dataset contains 15,408 games / 500,122 fixed-minute states through September 2. The deployed GAM remains `wpgam_v8_recency_series`, SHA-256 `fed34df9a7831a1d11c83d81a4b53f63661f108338890af737dbcda085c46589`. The v9 Elder, physical-gold and nested-calibration repairs are already implemented and staged. Recommending them as unfinished source work would repeat the earlier audit. See the [repair report](/Users/itch/Documents/Github/esports2/docs/model-repairs-2026-09-04.md).

**The most consequential new finding is historical HP lookahead.** The feed importer retains its opening timestamp only in memory, then stores the last frame of each requested window. The dataset builder subsequently treats the earliest *stored* timestamp as clock zero. These timestamps can differ. See [feed ingestion](/Users/itch/Documents/Github/esports2/scripts/feed_backfill.py:101), [last-frame storage](/Users/itch/Documents/Github/esports2/scripts/feed_backfill.py:116), and [dataset reanchoring](/Users/itch/Documents/Github/esports2/lol_ticker/wpx.py:513).

For game **71282**, played August 20, 2025, the archived opening timestamp is `1755733649`; the first stored observation is `1755733769`, labeled minute 2. The cached fixed-minute **t=0** state therefore has `has_hp=1` and `hp_pool=-0.10` from **120 seconds after the original feed origin**. Game 81541 shows a 16-second backward shift. Two of 2,609 linked historical feed games lack minute zero. This is not confined to missing opening minutes: five of a deterministic twelve-game development sample with minute zero present had origin offsets of 1–8 seconds.

This establishes a causality defect relative to the feed's original time origin. It does **not** quantify the direction or magnitude of bias in Brier scores, certify the feed origin as the true pause-adjusted game clock, or prove that a different model is better. Existing comparisons remain useful records of what was tried, but their causal interpretation and small ranking differences need rechecking after correction. The [evidence](/Users/itch/Documents/Github/esports2/docs/research-2026-09-11/hp_clock_evidence.json) and [read-only reproducer](/Users/itch/Documents/Github/esports2/docs/research-2026-09-11/reproduce_hp_clock.py) preserve the concrete cases. The reproducer selects input metadata, restricts dates to consumed history and verifies a read-only database transaction.

Persist the original feed origin and observation timestamps, then derive an explicit pause/attempt-aware clock. Before the first valid observation, use the documented missing-data fallback. Enforce `observation_clock <= prediction_clock` for every joined state, and verify that changing future observations cannot change earlier features. Version the corrected dataset and regenerate affected priors, transforms, calibration and models. Rerun the v8 reference architecture, repaired core and existing rich diagnostic control on identical corrected rows and date splits before expanding the search.

Several additional input improvements deserve bounded ablations after that correction:

| Opportunity | Observed evidence | Concrete next change |
|---|---|---|
| Telemetry quality | HP appears on 82,318 / 500,122 fixed rows (16.46%). Historical values can be carried for 90 seconds; live deaths use the current frame. | Store observation age/source; compare on a clock-corrected official-feed subset. Reconstruct per-player death state instead of counting recent kills. |
| Death reconstruction | 52 pre-May fixed rows exceed five deaths on one side before model preprocessing. GAM already clips these counts to five. | Fix victim-level reconstruction and represent estimation uncertainty; clipping is already present and does not remove measurement error. |
| Item parity | Historical accumulation includes purchases/sales but ignores destruction/undo. February–April data contains 568,018 destruction and 15,764 undo events. Live scoring sums current inventory. | Replay inventory transitions consistently and use patch-specific prices, or ablate the inconsistent channels. |
| Prior provenance | 177 / 2,075 February–April development games lack an OE link. Missing prior channels can be indistinguishable from equal ratings. | Add explicit availability, rating age, resolved-player count and rating-source roster metadata; shrink unreliable priors toward an appropriate fallback. |

The item issue affects the champion-state learner's nuisance regressors and legacy comparator; items are **not direct inputs to the main GAM state**. Historical prices come from the global item table, whose stored versions are predominantly 16.16.1, rather than the game's patch. Relevant paths are [historical items](/Users/itch/Documents/Github/esports2/lol_ticker/wpx.py:332), [accumulation](/Users/itch/Documents/Github/esports2/lol_ticker/wpx.py:450), [live inventory](/Users/itch/Documents/Github/esports2/lol_ticker/live.py:830), and [champion-state fitting](/Users/itch/Documents/Github/esports2/lol_ticker/wpgam.py:438). Riot provides versioned static data for a patch-aware implementation. [Riot Data Dragon documentation](https://developer.riotgames.com/docs/lol#data-dragon).

A further code-derived roster scenario needs a regression case: player priors describe the latest OE lineup, while roster-change detection compares with the latest gol.gg lineup. If gol.gg has already seen a substitution and OE lags, an apparently unchanged live lineup can retain the older player prior. Its frequency has not been measured. Compare against the actual rating-source roster. See [roster comparison](/Users/itch/Documents/Github/esports2/lol_ticker/live.py:666) and [adjustment decision](/Users/itch/Documents/Github/esports2/lol_ticker/live.py:762).

The existing experiments substantially narrow the useful model search. In the repaired monthly replay, lower game-balanced Brier is better:

| Existing method | Replay Brier | Interpretation |
|---|---:|---|
| Repaired core GAM | 0.143596 | Reference within this protocol |
| Rich champion/patch GAM | 0.143160 | Small uncertain gain; loses development and frozen comparisons |
| Monotone ensemble | 0.143461 | Small uncertain gain; frozen result worse |
| Monotone spline GAM | 0.143777 | No aggregate gain |
| Neural network ensemble | 0.144661 | No aggregate gain; sampled gold reversals |
| Monotone LightGBM | 0.146105 | No aggregate gain |
| Rich LightGBM | 0.150320 | Substantially worse in these configurations |

The repaired 180-day recency experiment has replay delta **+0.000145**, interval **[−0.000695, +0.001013]**. Major-only core training is **+0.001807** worse on the same 861 major-league games, interval **[−0.000675, +0.004223]**. Neither supports adoption. Rich champion×gold×time and patch terms have already been tested; early documents proposing them as untested are superseded by the [alternative-method suite](/Users/itch/Documents/Github/esports2/docs/alternative-methods-2026-09-04.md) and [major-league comparison](/Users/itch/Documents/Github/esports2/docs/major-league-comparison-2026-09-05.md). These conclusions apply to recorded configurations, not every possible tree or neural architecture.

New saved-prediction diagnostics further weaken a blanket rich-GAM recommendation. All rows below compare the same all-league-trained rich and core predictions. Differences are rich minus core; intervals resample whole date/match clusters with equal total weight per game.

| Scored subset | Games | Core Brier | Rich Brier | Difference [95% interval] |
|---|---:|---:|---:|---|
| Entire monthly replay | 3,097 | 0.143596 | 0.143160 | −0.000436 [−0.001688, +0.000776] |
| Excluding May's scored rows | 1,965 | 0.144933 | 0.145155 | +0.000222 [−0.001459, +0.001888] |
| Major leagues | 861 | 0.152380 | 0.152505 | +0.000124 [−0.002363, +0.002580] |
| Nonmajor leagues | 2,236 | 0.140213 | 0.139562 | −0.000652 [−0.002054, +0.000755] |
| OE prior missing-or-even proxy | 360 | 0.153140 | 0.155185 | +0.002045 [−0.001536, +0.005813] |

The May sensitivity removes scored rows; it does not refit models without May. The OE category is a zero-valued-feature proxy, not verified missingness. Every interval spans zero, and these exploratory slices are not adjusted for multiple inspection. They identify fragility, not proven subgroup effects. The [diagnostics and hashes](/Users/itch/Documents/Github/esports2/docs/research-2026-09-11/diagnostics.json) and [reproduction script](/Users/itch/Documents/Github/esports2/docs/research-2026-09-11/diagnostics.py) also verify all sixteen saved aggregate scores to numerical tolerance.

**The next modeling experiment should change the parameterization, not simply add more rich features.** The current [rich GAM](/Users/itch/Documents/Github/esports2/lol_ticker/wpadapt.py:130) builds separate category blocks for ten side/role slots, with nonnegative champion gold and gold×linear-time additions. Resource, patch and competition terms are bundled; the repaired search tries only two context penalties.

Start with a shared role resource curve and strongly shrunk champion deviations, pooling blue/red observations while retaining a distinct blue-side effect. Allow champion deviations above or below the role curve, but constrain the **total response to a physical gold increase** to remain nonnegative. Use training-only normalization and a small smooth time basis. Constraints must cover the composed transformations and clipping boundaries; positivity of isolated columns is insufficient. First compare constant resource deviations, then time-varying deviations, with a small prespecified shrinkage grid. Keep new patch dynamics and competition slopes out of this initial experiment.

Hierarchical GAMs and identifiable factor-smooth deviations provide the methodological basis for sharing curves and reducing variance. They motivate this hypothesis without guaranteeing an esports accuracy gain. [Pedersen et al., hierarchical GAMs](https://pmc.ncbi.nlm.nih.gov/articles/PMC6542350/), [official mgcv factor-smooth documentation](https://www.stat.ethz.ch/R-manual/R-devel/library/mgcv/html/factor.smooth.html).

Only if corrected development results show persistent drift should a second experiment adapt a few coefficients across observed patches. Begin with zero drift versus a small intercept/champion deviation update, retaining long-history objective and state knowledge. This differs from global recency weighting, which discounts every relationship together. Likewise, competition-specific effects should borrow from a common curve rather than discard lower-tier games. Dynamic generalized linear models supply a principled framework for evolving a limited state. [West, Harrison and Migon](https://www2.stat.duke.edu/homeweb/mw/MWextrapubs/West1985a.pdf). These remain lower-priority hypotheses.

**Prospective evaluation needs a candidate path.** Architectural staging correctly protects v8, but [shadow scoring](/Users/itch/Documents/Github/esports2/lol_ticker/shadow.py:381) uses the deployed forecast; repaired/rich candidates do not receive immutable forecasts on those same frames. Add candidate forecasts keyed to the exact captured input with artifact hash, input-contract hash, source revision and training cutoff. Evaluate incumbent versus one frozen challenger on all mutually valid model frames. Exchange quote coverage should govern the separate market comparison, not unnecessarily discard model-versus-model evidence. Previously captured outcomes cannot retroactively make a newly selected challenger prospective.

The latest saved v12 report is **September 11, 2026, 22:57 PDT / September 12, 05:57 UTC**. It supersedes the September 4 snapshots:

| Primary live comparison | Kalshi | Polymarket |
|---|---:|---:|
| Eligible games / states | 73 / 858 | 65 / 1,127 |
| Incumbent model Brier | 0.154313 | 0.173191 |
| Market Brier | 0.141486 | 0.156722 |
| Market minus model, 95% interval | −0.012827 [−0.032693, +0.006612] | −0.016469 [−0.035160, +0.001432] |
| Eligible fraction of resolved model rows | 22.3% | 29.2% |
| Remaining to existing 100-game target | 27 | 35 |

Both point estimates favor the markets; both intervals span zero. These are deployed-v8 results, not candidate evidence, and platform game counts may overlap. Median upstream feed age is now **58.951 seconds**, with **0.736 seconds** processing after receipt. A median cannot establish tail freshness. Preserve the registered 90-second lead limit and report p95/p99 plus failures by platform, league, phase and quote-cache reason. The [saved snapshot](/Users/itch/Documents/Github/esports2/docs/research-2026-09-11/shadow_snapshot.json) retains the exact report timestamp and source hash.

The quote path is already bounded and asynchronous. Next investigate independent refresh of known market IDs and per-game deadlines for feed fetching: [live.py](/Users/itch/Documents/Github/esports2/lol_ticker/live.py:1088) can search eighteen windows, while games are handled serially by the recorder. A stalled feed can therefore delay other games. Idempotence of repeated windows deserves a targeted test; it is a code-inspection concern, not a demonstrated scoring defect. Licensed direct telemetry is an infrastructure option if coverage and latency remain limiting: Riot identifies GRID as its official data partner, and GRID offers commercial LoL feeds. Access, coverage, terms and measured delay would require separate evaluation; no service was purchased or contacted. [Riot official esports data](https://riotesportsdata.com/en-us/), [GRID LoL data](https://grid.gg/data/games/league-of-legends/), [GRID access](https://grid.gg/get-access/).

Three evaluation details should be resolved before a promotion experiment. First, the live registry is consumed through September 2 but the historical-odds registry stops at August 24. The latter's [date-based eligibility logic](/Users/itch/Documents/Github/esports2/lol_ticker/wphist.py:237) would count **131 already-inspected games** as fresh once supplied a compatible v9 base. The current default v8 base is rejected before this path, and this audit did not establish that reuse actually happened. Maintain a shared exposure inventory and experiment-specific frozen plans so a different registry cannot relabel inspected outcomes as fresh.

Second, [the deployment gate](/Users/itch/Documents/Github/esports2/lol_ticker/wpdeploy.py:55) resamples games and compares a blend against its newly fitted GAM. It needs a direct challenger-versus-incumbent comparison, series/date uncertainty, and provenance binding for precomputed results. The same-kind refresh path can replace the GAM; changes to training or calibration deserve evaluation even without a new model-kind string. Third, [the older benchmark](/Users/itch/Documents/Github/esports2/lol_ticker/wpbench.py:446) still recenters its intercept on fitted training predictions. Repaired production and the newer methods runner retain held-out calibration coefficients. Consolidate this logic; the mismatch is not a claim of remaining calibration leakage in the repaired methods suite. Independent fitting/calibration data is the relevant methodological requirement. [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

The next iteration should follow these decision points:

1. **Correct and audit the inputs.** Require zero future timestamp joins, documented pre-observation fallback, coherent inventory semantics where used, and versioned rebuilt artifacts. Verify causality on the full rebuilt corpus, including pauses and incomplete openings.
2. **Reestablish reference scores.** Refit the reference architectures on identical corrected rows, preserving historical split boundaries and chronological stacking/calibration. These consumed dates remain development evidence. A correctness repair and an accuracy gain are separate claims.
3. **Run one narrow resource experiment.** Select its shrinkage and complexity on earlier blocks. If benefits are uncertain or depend on a single block, prefer the simpler core. Audit physical gold behavior on every monthly artifact, plus Elder/death and missing-input slices.
4. **Freeze one candidate and score forward.** Prespecify target population, game-balanced Brier, minimum worthwhile gain or tolerated regression, supporting log loss/calibration checks, and a fixed endpoint or valid sequential rule. Determine sample size from corrected paired series-loss variability and the chosen precision target. The existing 100-game threshold is a collection floor, not proof of sufficient statistical power.

Research artifacts are confined to this report and [its evidence directory](/Users/itch/Documents/Github/esports2/docs/research-2026-09-11). The saved-prediction diagnostic validates dataset/resource hashes, probability bounds, cohort alignment and all sixteen reported Brier/log-loss values; it confirms the deployed model, manifest, dataset and live registry hashes are unchanged during execution. The HP reproducer was executed successfully by the data-audit agent. It reproduces the two late-start examples and linked-feed counts; the supplementary twelve-game sample and item/death aggregates are recorded audit observations, not regenerated by that script. This research does not claim that any proposed change has already improved accuracy.

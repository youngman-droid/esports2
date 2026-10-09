# TGV public response follow-up

The resumed investigation made progress; the alternative-source objective remains open.

## Current deployment

The homepage now references `index-r34Uubfu.js`. Recursively followed 59 linked JavaScript assets; saved bodies and hashes in `data/tgv/20260916/current-surface/`. Earlier deployment URLs can return HTTP 200 HTML, so the historical-attempt inventory labels those fallbacks explicitly. Current crawl rejects HTML.

The current client still exposes the manifest-backed catalog, SQ strengths, champion relations, rankings, skill history, and draft gaps. PredictionsQuery and SystemStatusQuery explicitly take access keys; custom draft also requires one. No restricted calls were attempted. Public arcade uses normal guest identity authentication and warrants further inspection of read-only operations. This is an inventory of observed client surfaces, not proof that every possible server response has been enumerated.

## A second public model version

[Fallback catalog](https://tgv-analytics.com/data/current-model-catalog.json) returned real JSON, SHA256 `432e692c04528ce7fbbfa8c1bb121f7dcea142eaba245b6bc5cfe5915ea0f35d`.

It contains model `adb3014f1f3dc1ae0d701cc0453d51b61dc691d87e99ff8b6640744ca6fa9152`, patch 16.17, generated September 3. The refreshed public challenge remains model `5b9316f481bb4280561dad70b4406e0daba9658053222296e91766e3dd88ba2b`. Keep these sources separate; never combine coefficients across versions. Next: inspect older catalog fields and the corresponding client-referenced static JSON paths for additional inputs and provenance.

## New numerical constraints from draft gaps

`research/tgv_gap_constraints.py` recovers the central score(s) from each team's published median score and median sigmoid lift. For an even sample, let the two central scores be m-d and m+d, and let L be the median probability minus one half. Then:

`cosh(d) = sinh(m)/(2L) - cosh(m)`.

For odd samples the central score is already the published median. Zero medians do not identify the spread; near-zero inversion is ill-conditioned. Recovered 545 central score values across 296 teams. Reconstructing their published probability lifts gives maximum absolute error 8.33e-17. Output: `data/tgv/20260916/gap-central-score-constraints.json`.

These are order-statistic constraints, not 545 independently identified drafts. Game identities, other scores, and source-history alignment remain unresolved. Next: match teams with short scored histories to local match data and evaluate whether these constraints reveal composition terms.

## Anonymous GraphQL model responses verified

Extracted the API schema embedded in the current public bundle and saved `public-api-schema.json` and `query-inventory.json`. Used the site's configured Cognito guest identity flow and signed read-only requests. Temporary guest credentials were held only in process memory; no game or score was created.

`ChampionModelQuery(patches: [String]!)` succeeds anonymously. For 16.18 its 865 role/champion strengths exactly match the full-feature static catalog (maximum difference zero). This confirms another working model-data route but adds no new coefficients in that response.

The `historyChampionId` and `historyRole` variant returns `modelVersion`, `championId`, `role`, and `points`, with `patch`, `modelPatch`, and `relativeStrength` for each point. Annie top on 16.18 gives -0.08107535889672879, compared with its full-feature unary -0.0053196386631905734. The relative-strength definition still needs investigation; these values must not be substituted for the full unary.

Requesting the older public catalog's patch 16.11 from this live query returned an explicit error: “Requested patch 16.11 predates immutable model coverage.” The older catalog does not unlock earlier patch coverage for the live model.

The static legacy runtime and one champion-relation JSON also exist and agree on the older model version. Five other client-named fallback JSON paths returned HTML rather than JSON. Audit saved in `legacy-static/response-audit.json`. The older catalog uses standardized-probit with scoreScale 1.138927594582921, unlike the current model's logits.

Remaining useful work: identify the history relative-strength transformation; inspect supported roster-query input semantics; assess normal guest review/leaderboard responses; match the three teams with exactly two scored games to local data (their paired medians recover both draft scores, without the order-statistic ambiguity of longer histories). No claim that all possible JSON responses are exhausted.

## Model variants resolved and six candidate draft constraints

The history transformation is exactly `full unary(champion) - full unary(Aatrox)` for both Annie and Olaf top across all seven supported patches (14 points, max error zero). This replaces the earlier unknown-transform finding for these tested cases. Other roles have not been tested. It provides no additional fitted coefficient in these responses.

The optional roster is a role-to-list-of-champion-IDs filter. `{top:[1,2], jungle:[78]}` returns those two top champions, one jungle champion, and all 173 champions for each omitted role. All returned strengths match the static full-feature unary exactly. This is not an API for player comfort.

Normal guest `EngineAccessQuery(mode:puzzle)` with operation `leaderboards` succeeds. Its response provides puzzle leaderboard metadata rather than draft assignments or model breakdowns. A read-only `counterfactual` for a publicly listed current choice failed with the generic “Puzzle request failed. Refresh and try again.” Repeating once with the normal client requestId and clientCacheHit fields also failed. No claim about the underlying cause is supported; no game was started or score submitted.

`research/tgv_two_game_constraints.py` matches the three teams with exactly two scored games to two local covered-patch games each, bounded by TGV's team-specific lastPlayedAt. Candidates are Only The Family (August 20, patch 16.16), Fortress Esports (July 15, patch 16.13), and Miðgarð Esports (August 28, patch 16.16). These are candidate source matches, not verified TGV game IDs. Only The Family has later local games not reflected in that TGV team record; they are excluded explicitly.

For each candidate draft, subtract independently recovered unary, matchup, and synergy from each possible summary score. The remaining value represents composition plus comfort in the team's perspective, assuming the source match and published gap method align. Both score assignments are retained for every team. Smaller residuals are not treated as proof of the correct assignment. Results are saved in `two-game-draft-constraints.json`.

Current remaining avenues include testing the schema's separate ArcadeLeaderboardQuery, checking any public archive links or embedded model JSON beyond the observed release, and extracting stronger constraints by aligning local historical comfort counts and team identities. Full-model reconstruction remains unproven.

## Additional public response routes

The separate `ArcadeLeaderboardQuery` returns “The classic leaderboard has been retired.” `EngineAccessQuery(mode:dailyChallenge)` succeeds anonymously and returns the same live model and challenge as REST. The 19 detected serialization differences are integer versus floating zero regret fields, not different values or extra parameters. The advertised sitemap lists only home, arcade, champions, teams, contact, privacy, and terms; no archive route was found there. Public web search did not locate a relevant model repository.

The exact daily-challenge-source storage key in the public release manifest returned HTTP 403 when requested at the public model-assets host. As with the factor source, a storage key does not confirm a public-serving path. No alternate credentials or access bypasses were attempted.

The manifest's endpoint parity certificate is publisher-supplied provenance, not a local reproduction test. In particular its semantic numeric values compared is zero; its stated semantic proof is checkpoint-bound structural identity and payoff probes. This does not expose the missing numeric composition inputs.

## Complete legacy relation collection

Downloaded and validated all 865 client-addressed legacy relation files, 43,971,492 bytes, zero errors. All report the older model version; all have the same seven top-level fields, with no additional composition payload. Saved hashes are local provenance, not publisher-provided checksums.

Compared 748,225 matchup cells, 595,120 synergy cells, and 5,190 overlapping unary cells with the live export. Old/new correlations are 0.122, 0.390, and 0.469 respectively. A fitted global affine transformation leaves maximum residuals 0.540, 0.089, and 1.442. The old model therefore cannot be treated as a simple rescaling of the current model. Its extra patches 16.10–16.11 do not supply those patches for the current model.

Reproduction: `python3 research/tgv_legacy_scrape.py`, then `python3 research/tgv_legacy_compare.py`. Both completed successfully. Full comparison statistics are in `data/tgv/20260916/legacy-static/comparison.json`.

## Conditional composition from historical comfort

Queried the local database in a read-only transaction for the 60 player/role/champion combinations in the six candidate drafts. Before-game and before-day counts agree for all 60. Applied the recovered saturating comfort curve, then subtracted the resulting team comfort difference from the unexplained draft residual. Also computed a separate export-time-count scenario using the saved September 14 aggregate counts.

The timing convention materially affects the inference: the first Only The Family draft has historical comfort -0.1250511 versus export-time comfort -0.0321384, a difference of 0.0929127 logits. The public gap metadata does not resolve this convention. Both score assignments and both timing scenarios are preserved in `candidate-composition-scenarios.json`; no conditional residual is labeled an independently recovered composition factor. The six candidate drafts are insufficient to identify the full composition tensor even if alignment and timing were resolved.

Reproduction: `python3 research/tgv_candidate_composition.py` with saved count inputs. The read-only SQL source is archived as `data/tgv/20260916/historical-comfort-source.py`.

At this point no additional model-data route has been identified in the downloaded current client, its embedded API schema, manifest, sitemap, or legacy static data. This is a bounded observed-surface inventory, not proof of every possible response on the server. Further exact reconstruction needs additional independent draft evaluations or authorized factor-export access; unrelated payment/account APIs and gated private records do not serve this objective.

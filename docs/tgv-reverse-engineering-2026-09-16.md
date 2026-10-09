# TGV model reconstruction from public data

We downloaded and checksum-verified **871 public JSON assets (50,519,618 bytes)** for model `5b9316f481bb4280561dad70b4406e0daba9658053222296e91766e3dd88ba2b`, prepared September 15, 2026. The release covers 173 champions, five roles and patches 16.12–16.18. This goes substantially beyond the earlier website comparison: the public data reveal the statistical family, draft factors, exact experience/comfort curves, and ranking decomposition.

The full model is not yet reproducible for arbitrary drafts. The public factor-source object returned HTTP 403, and the custom-draft client requires an access key. No access controls were bypassed. In particular, per-champion composition inputs and several composition parameters remain unavailable. A reconstruction that silently sets these to zero would be misleading.

## Evidence and recovered data

The public [daily challenge endpoint](https://draft-api.tgv-analytics.com/v1/challenge) returns a versioned asset URL and a pinned SHA-256 for its [release manifest](https://model-assets.tgv-analytics.com/releases/5b9316f481bb4280561dad70b4406e0daba9658053222296e91766e3dd88ba2b/assets/release-manifest.json). The scraper downloads only JSON files explicitly listed in that manifest, verifies bytes and SHA-256, skips media and resumes already verified files. `robots.txt` allows crawling.

| Recovered data | Coverage |
|---|---:|
| Full-model champion/role/patch strengths | 6,055 values |
| Solo-queue-only champion/role/patch strengths | 6,055 values |
| Directed matchup entries | 748,225 |
| Directed synergy entries | 595,120 |
| Team ranking decompositions | 968 teams / 4,840 roster rows |
| Team draft-gap summaries | 968 teams |
| Published checksum draft scores | 56 |
| Complete daily terminal-draft breakdown | 1 |

Matchups are exactly antisymmetric under swapping both role/champion endpoints; synergy entries are exactly symmetric. Synergy rows omit same-role and duplicate-champion pairs, which are not legal within a completed five-role draft. They are retained as missing values in the extracted tensor rather than silently fabricated as zeros.

Published provenance in the [catalog](https://model-assets.tgv-analytics.com/releases/5b9316f481bb4280561dad70b4406e0daba9658053222296e91766e3dd88ba2b/assets/current-model-catalog.json) identifies both solo-queue and professional likelihoods as `bernoulli_logistic`, solo-queue interactions as `canonical_sparse`, and transfer policy as `four_positive_global_scales_plus_two_outcome_weights`. The exact assignment of those six transfer parameters is not disclosed. Seven patch counts total **6,926,321 solo-queue games**; these are published coverage counts, not an independently audited training population. The release contains split/checkpoint hashes, but no readable training split or predictive benchmark.

## Draft scoring formula

For role-ordered blue champions b and red champions r:

```
z = side_bias(patch)
    + sum_role [U(patch, role, b) - U(patch, role, r)]
    + sum_role [comfort_blue - comfort_red]
    + sum_blue_role,sum_red_role M(blue_role, red_role, b, r)
    + sum_role_i<role_j [S(blue_i, blue_j) - S(red_i, red_j)]
    + composition(blue, red, patch)

P(blue) = 1 / (1 + exp(-z))
```

The browser supports older standardized-probit releases, but **this release is logit**. Field names containing `Probit` persist and should not be used to infer the current link. The UI saturates its logistic display outside ±8.

The [champion board strength file](https://model-assets.tgv-analytics.com/releases/5b9316f481bb4280561dad70b4406e0daba9658053222296e91766e3dd88ba2b/assets/social-betting-strengths.v1.json) explicitly uses `sq_only_betting_v1`. The browser replaces the catalog's full-model strengths with those values for its headline champion board. The draft engine uses the full model. **The visible board STR values are therefore not interchangeable with full draft-engine unary effects.** Display formatting multiplies coefficients by 1,000 and rounds: +130 means approximately +0.130 score units, not 130 Elo or 13 percentage points.

Independent reconstruction of the daily terminal draft:

| Component | Blue log-odds contribution | Evidence |
|---|---:|---|
| Champion strength | −0.07828534321588368 | Recomputed from full catalog |
| Comfort | +0.021454843215883646 | Recomputed from published pool modifiers |
| Matchups | +0.06748132508030077 | Recomputed from 25 table entries |
| Synergy | −0.0059911176132110755 | Recomputed from 20 table entries |
| Side | +0.12602036532208627 | Supplied by reference breakdown |
| Composition | −0.011848672936393218 | Supplied by reference breakdown |
| **Total** | **+0.11883139985278271** | **52.96729407884517% blue** |

The first four factors match the independently published breakdown exactly in the saved calculation. The total matches its double-precision review exactly; the search-engine value differs by 1.12e-8 log odds, consistent with numerical precision differences. This is **not independent verification of side or composition**, which were supplied from the reference.

Per-champion attribution is also not a raw champion coefficient: the reference allocates half of a same-role blue-minus-red strength difference to each champion. Reading those per-champion `modelStrength` values as standalone strengths would be wrong.

## Recovered player comfort and experience

The [ranking data](https://model-assets.tgv-analytics.com/releases/5b9316f481bb4280561dad70b4406e0daba9658053222296e91766e3dd88ba2b/assets/team-rankings.v3.json) reveal simple saturating count functions. Parameters were solved from one-game and two-game values; higher counts provide checks.

```
experience(N) = 0.7797376989690219 * N / (N + 68.72921665596814)

champion_comfort(n) = 0.07896877604172117 * n / (n + 1.036248079221949)

roster_comfort(player, role)
    = sum_champion [n_champion / N * champion_comfort(n_champion)]
```

Here n is cumulative professional appearances on that champion in that role, and N is total appearances in that role. The parameters are release-specific. The roster comfort formula is an inference supported by numerical reconstruction, not an exported training implementation.

- Experience matches **all 4,840 roster rows**, maximum error 3.69e-13.
- Our local appearance totals match **3,158 roster rows / 2,348 distinct player-role identities**.
- The comfort formula matches **3,156 of those 3,158 rows** within 1e-10, including **3,031 of 3,033 rows with more than two appearances**.
- The two exceptions both represent Decay mid. Their maximum error is 0.0001974 log odds. Equal total counts do not prove identical champion allocation; this is an unresolved source mismatch, not a reason to adjust the formula to fit those rows.

This is a **familiarity bonus**, not our earlier win-rate-based mastery proxy. It saturates rapidly: the first few professional appearances account for much of the available bonus. Outcome residuals or GD@15 are not required to reproduce the published roster comfort component. This does not prove that no other player/champion performance information exists elsewhere in TGV's model.

Daily-puzzle modifiers are generated values and may be negative or exceed this professional familiarity curve. They enter additively but must not be used to infer the professional comfort estimator.

## Team skill and ranking

Every published ranking reconciles as:

```
player_skill = player_strength + league_mix + experience
               + champion_strength + comfort
team_skill = team_strength + sum_five_players(player_skill)
displayed_Elo = 1200 + 400 / ln(10) * team_skill
```

All 968 Elo values reproduce exactly, and component sums agree within 8.9e-16. Thus the displayed Elo is a conversion of an additive skill model; the public data do not establish that it is updated by conventional Elo's K-factor rule.

The roster champion-strength component equals a pick-frequency-weighted average of current-patch solo-queue-only unary strengths plus a role constant. Recovered constants: top 1.172683905842013; jungle 1.8161161859324937; middle 0.7877564575116629; bottom −0.055025702884713035; support 1.0245779917636488. All matching-history identities except the same Decay discrepancy reconcile to numerical precision. These role constants cancel when comparing two complete five-role rosters.

The published `team-draft-gaps.v2.json` describes original-patch, actual-player-comfort, last-ten-map median draft scores, with side bias removed. Its probability lift is a median of sigmoid(score)−0.5; it should not be assumed to equal sigmoid(median score)−0.5 exactly for an even number of maps.

## Composition: recovered structure, incomplete parameters

The public draft-review client defines:

- Gold demand as the sum of modeled champion gold shares, not actual gold at minute 10.
- Tankiness as modeled damage taken plus self-mitigation per minute and per life.
- Total damage as magic + physical + true damage per minute.
- Damage balance as squared distance of log(magic/physical) from a learned center, with its coefficient adjusted by opposing normalized tankiness.

The terminal breakdown permits these effective linear coefficients to be recovered by dividing effect by blue-minus-red feature value:

| Feature | Effective coefficient |
|---|---:|
| True DPM | +0.0002266157858898593 |
| Tankiness | +0.00008286550292128596 |
| Gold demand | −0.10104144346579348 |
| Total DPM | −0.00012201848761611174 |

These are inferred from one published terminal example, not independently established as global constants across all patches and contexts. The negative total-DPM coefficient is conditional on the other composition terms; it is not a causal claim that dealing less damage helps win.

The manifest identifies a composition-input tensor shaped [7 patches, 5 roles, 173 champions, 7 channels], but only its hash/shape are public in the downloaded assets. The champion-level inputs, balance center, enemy-tankiness normalization/interaction parameters, and six historical patch side biases remain unresolved. Summing recovered unary/pair terms does **not** reproduce the 56 published checksum draft scores: the maximum residual using SQ unary + matchup + synergy is 0.334999 log odds. This is an explicit failed full-replication check, not a passing test hidden by the one terminal example.

A simple cumulative-league-mixture linear hypothesis was also tested against 2,348 matched player-role rows. It has RMSE 0.012206 and maximum error 0.155368, so those fitted league weights are **not** an exact reconstruction. Professional player/team updates, league-history weighting and training penalties remain unknown.

### Follow-up: league history across roles

Further inspection showed that the league component can be identical across different roles held by the same player. Pooling league appearances across **all roles**, then applying a small denominator offset, explains most of the earlier discrepancy:

```
league_mix(player) ≈ sum_league [games(player, league) * league_weight]
                     / (all_role_games(player) + 0.6900001310809816)
```

The denominator offset was recovered from pure-NACL one-game and two-game examples. League weights were fitted on 1,874 distinct players with robust least squares; a deterministic player-identity split reserved 435 players. All 68 league columns are identified by the fitting matrix, including the exposure combinations in the held-out set.

**418/435 held-out players (96.1%)** reproduce within 1e-6, with median absolute error 1.14e-8. The maximum held-out error remains 0.020552, so this is strong evidence for the structure, not an exact recovery of every row. Matching current-role totals does not certify that our source contains the same complete all-role history. The 17 held-out exceptions are preserved. This reconstruction is of published component values; these are not held-out match-outcome accuracy results.

The result, parameters, exceptions and hashes are in `league-reconstruction.json`. Reproduce with `OPENBLAS_NUM_THREADS=1 python3 scripts/tgv_league_recover.py`. This supersedes the earlier role-specific linear hypothesis but does not establish TGV's training/update procedure.

### Numerical coverage of the missing composition

An audit of the 56 checksum drafts and one published terminal example finds 548 distinct patch/role/champion cells represented, leaving **5,507 of 6,055 cells unobserved** in those terminal constraints. Even a simplified missing additive coefficient per cell has only 57 independent equations and 5,998 unconstrained dimensions. The actual nonlinear composition model requires additional information. This is a numerical coverage argument: cryptographic hashes identify a checkpoint but do not expose its numbers, and puzzle action values without their counterfactual terminal assignments cannot be treated as direct composition observations.

The reproducible coverage audit is in `composition-identifiability-audit.json`, generated by `scripts/tgv_analyze.py`. It rules out claiming a general exact reconstruction merely by fitting the available terminal examples.

## What this changes about our comparison

TGV and our isolated draft scorer both have additive logistic draft factors, but TGV exposes a much larger solo-queue-derived champion foundation, explicit role-pair interactions, nonlinear composition, and a precisely defined count-based comfort bonus. Our prior experiment's relative-win-rate and log-experience features do not replicate their comfort function. Its inconclusive gain therefore does not rule out TGV's particular estimator.

Conversely, recovering coefficients and numerical identities says nothing about which model predicts unseen professional games better. The released model was prepared in September; applying it to earlier matches would not produce an out-of-sample comparison. Training splits are represented only by hashes, and no predictive calibration/accuracy benchmark was recovered.

## Reproduction and remaining work

Local artifacts are under `data/tgv/20260916/` (ignored research data): original public assets, scrape manifest, public client modules, packed `recovered-factors.npz`, `reconstruction-report.json`, `daily-reconstruction.json`, local count snapshots, curve estimates, and explicit failed checksum/league-mixture diagnostics.

Code:

- `scripts/tgv_scrape.py`: resume and verify the public JSON scrape.
- `scripts/tgv_analyze.py`: verify all asset hashes, recover curves and tensor properties, check rankings and reference scores.
- `lol_ticker/tgv_reconstruction.py`: isolated factor scorer. Full probability requires explicit composition and side effects; missing terms never default to zero.
- `tests/test_tgv_reconstruction.py`: curve recovery, logistic link, published factor parity and refusal to fabricate missing terms.

```sh
python3 scripts/tgv_scrape.py
python3 scripts/tgv_analyze.py
python3 -m unittest discover -s tests -p test_tgv_reconstruction.py -v
```

Completion of a general exact scorer still requires either authorized access to the full factor export or enough authorized custom-draft evaluations to identify the missing composition functions. The public scrape and verified partial reconstruction are complete; the full arbitrary-draft replica is not. Production models and database contents are unchanged.

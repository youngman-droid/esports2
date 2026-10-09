# Composition and Fearless draft context — October 4, 2026

Implemented separate patch-bound composition and certified-series Fearless
feature/candidate modules. The composition screen retained the existing draft
baseline. Fearless has no supported historical rule/series certificates in the
saved draft snapshot, so its predictive value remains unevaluated. No forecast
model was promoted and no evaluation registration was changed.

## Composition sources, capture and limits

`lol_ticker/wpcomposition.py` captures twelve blue-minus-red completed-draft
features. Four literal Data Dragon proxies are currently supported: Tank-tag
count, mean basic-attack range, summed HP growth and summed armor growth. They
describe source fields; they do not claim measured frontline effectiveness,
spell reach, or overall late-game strength.

The other eight feature columns require explicit reviewed capability records:
engage, disengage, waveclear and late-scaling counts; physical, magic and mixed
primary-damage counts; and a categorical damage-balance proxy. Capability
records need an exact static version, named reviewer, reviewed status,
primary-source evidence URLs and an evidence-availability timestamp. The
damage proxy gives every champion equal weight and splits a mixed label in
half; it does not estimate real dealt-damage shares. No capability labels or
numerical strengths were invented to fill gaps. **All eight reviewed columns
have zero coverage in the completed historical screen.**

Every feature family requires ten complete champion records; an incomplete
family is zero while other valid families remain usable. Role/participant
metadata is joined by ID and verified roles, not list order. Missing patch,
future source availability, wrong source version and invalid records fail
closed. Malformed optional stats/traits also remain zero and JSON-safe.

`scripts/composition_catalog.py` downloaded 65 immutable Data Dragon releases
used by the already-consumed draft archive, plus current releases 16.18.1 and
16.19.1. Exact response bytes, source hashes, representation hashes and HTTP
Last-Modified timestamps are retained under `data/composition/`. The timestamp
is a conservative bound for availability of the returned representation, not a
guessed patch-release date. Historical scoring uses UTC-day opening as its
source gate because the saved draft archive has dates rather than draft-second
timestamps. A major/minor game patch resolves to the saved `.1` catalog with
that precision limitation recorded; it does not certify the game's hotfix.

One actual source hazard is handled explicitly: patch 16.15.1's summary also
contains `Jade_*` League Classic champions sharing standard champion display
names. Those mode records are excluded before the name join, preventing them
from overwriting Summoner's Rift stats. This is covered by a regression test.

Riot documents versioned champion summaries and individual ability files in
its [Data Dragon documentation](https://developer.riotgames.com/docs/lol#data-dragon).
They are useful sources for future capability review; Tank tags alone cannot
supply engage, disengage, waveclear or damage-profile labels.

## Fearless rules, series history and observed pools

`lol_ticker/wpfearless.py` captures twenty role-specific pool advantages:
remaining supported champion counts, retained fractions, selected-pick support
and unseen-pick advantages. A supported champion needs three earlier role
appearances; each player-role history needs ten appearances. The counts measure
observed experience, not mastery, champion strength or winning performance.
Remaining pools apply the Fearless restriction only; current standard bans and
disabled champions are separate inputs.

Rules are explicit profiles scoped to an exact tournament, phase and effective
date window, with a primary source and a verified status. There is no league-name
default. The implementation handles both-team exclusion, own-team exclusion,
standard draft and explicit reset games. This flexibility does not claim all
variants apply to any particular league. The supplied First Stand 2025 profiles
uses Riot's [event announcement](https://lolesports.com/en-US/news/lol-esports-in-2025),
which describes prior picks becoming unavailable to both teams throughout the
series. Round-robin and knockout profiles preserve their distinct Bo3/Bo5
limits. They are confined to that event; exact source IDs and phases must still
be certified.

Series history must be a complete contiguous earlier-game prefix with stable
team IDs and completion timestamps preceding the current draft. Side swaps do
not change team identity. Wrong series, gaps, future games, illegal repeat picks
or a conflicting current draft make the block unavailable. Certified champion
removals remain visible as `series_available=true` when player history is
missing, while every fitted pool feature stays zero. Missing player-role
support independently disables that role pair.

`scripts/fearless_pools.py` built an offline snapshot from the consumed archive:
97,240 earlier appearances in its fixed 365-day window, zero rejected records,
with date-only timing represented conservatively. The maximum input outcome
date is September 2, 2026. IDs remain in the Oracle's Elixir namespace. The
snapshot explicitly declares that no Riot esports-player-ID bridge exists;
it is not installed as the live default. Live capture requires a pool snapshot
in the `riot_esports_player_id` namespace.

The saved 28,463-map draft archive lacks certified series IDs, exact phases and
matched rule profiles. Its Fearless feature coverage is therefore zero. This
is a source-coverage limitation, not evidence that Fearless features perform
poorly. A certified per-series bundle can be supplied without changing the
module or fetching new outcomes.

## Historical screen and reproducibility

`scripts/wpx_draft_context.py` reproduces the saved player-controlled draft
baseline's validation and later metrics before fitting any residual. It uses
28,463 already-consumed maps from January 1, 2024 through September 2, 2026:
20,332 training games before January 16, 3,773 validation games before May 1,
and 4,358 later diagnostic games. RMS scaling uses training data only; sources
are gated by patch and earlier availability time. The frozen baseline is never
refit; missing extras return its odds
exactly. Training offsets are the baseline's original in-sample pre-January
predictions, **not independent out-of-fold predictions**; this is an exploratory
screen, not a replacement evaluation against the deployed live model.

Composition uses signed ridge coefficients; Fearless uses positive
pool-advantage coefficients. Both isolated draft-time residuals use fixed ridge
800 and smooth penalty 70, no intercept, no new probability calibration and
no production dispatch. Validation requires a negative upper paired Brier
bound and no log-loss regression; otherwise selection retains baseline.

| Slice | Baseline Brier | Composition Brier | Change [95% date/match-cluster interval] |
|---|---:|---:|---:|
| Validation | 0.20765707 | 0.20767942 | +0.00002236 [−0.00008090, +0.00012255] |
| Later consumed diagnostic | 0.21179088 | 0.21181981 | +0.00002892 [−0.00005728, +0.00012142] |

Static composition coverage is 28,422/28,463 games (99.86%); full reviewed
composition coverage is zero. The four proxy residual fails validation
selection and slightly worsens log loss in both slices. This result applies to
these four static proxies and this residual formulation, not to a future fully
reviewed composition model.

The final completed run is `data/wpx/draft_context_2026-10-04_final_profiles/`. It retains
models, paired predictions, coverage, source snapshots and completion hashes.
All scoring dependencies are bound before fitting and verified unchanged at
completion. Earlier runs remain separate. The data/catalog/candidate paths are
isolated from deployed artifacts and frozen registrations.

```sh
python3 scripts/composition_catalog.py \
  --consumed-input data/wpx/draft_comfort_comparison_20260916/inputs.json
python3 scripts/fearless_pools.py --out data/fearless/offline_pool_reproduction.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 scripts/wpx_draft_context.py --out data/wpx/draft_context_reproduction
python3 -m unittest tests.test_wpcomposition tests.test_wpfearless \
  tests.test_composition_catalog tests.test_fearless_pools
```

## Integration APIs

Composition capture:

```python
wpcomposition.capture(metadata, patch, as_of_ts=draft_ts)
# Local data/composition/{catalogs,annotations}/VERSION.json; no network fetch.
wpcomposition.from_metadata(metadata, patch, catalog=catalog,
                            annotations=annotations, as_of_ts=draft_ts)
```

Fearless capture:

```python
wpfearless.capture(metadata, {"series_id": series_id, "game_num": game_num},
                   as_of_ts=draft_ts)
# Local data/fearless/series/ID.json must carry context/prior_games/rules.
# Optional default player history: data/fearless/player_pools.json.
wpfearless.capture(metadata, certified_bundle, as_of_ts=draft_ts)
```

Both modules expose `fit(extra, baseline_p, y, gids, spec=None)`,
`predict(model, extra, baseline_p)`, `save(model, path)` and `load(path)`.
Direct source adapters may use `wpcomposition.features(...)`,
`wpfearless.features(...)` and `wpfearless.build_player_pools(...)`.
The recorder can retain these JSON blocks while forecast serving remains on its
existing baseline. Capability review, additional tournament rule certificates
and a verified player-ID bridge remain the next source tasks.

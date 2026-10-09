# Confidence-aware ratings and an early solo-queue channel — October 4, 2026

Two isolated candidates are implemented. The confidence policy is ready for
certified live metadata but has no valid retrospective accuracy measurement.
The early solo-queue screen selected the unchanged corrected baseline. Neither
candidate was promoted, registered or added to production inference.

## Confidence-aware rating candidate

`lol_ticker/wpconfidence.py` exposes `capture(priors, as_of_ts=..., teams=...)`.
It preserves the original input and returns JSON metadata plus an independent
`adjusted_priors` candidate. Per-side Elo/player Elo values shrink around the
same 1500 anchor before the blue-minus-red differential is recomputed. The
adapter refuses a differential inconsistent with its per-side rating values.

The prespecified policy uses a 60-day age half-life, support factor
`n/(n+20)` when an actual count is supplied, and the verified matched fraction
of the exact rating-source roster. Shrink strength is 0.5. These are explicit
hypothesis settings, not parameters selected by observed accuracy. Unknown
support or continuity remains visibly unknown; it is not fabricated as zero
games or zero continuity. Missing source coverage, missing dates, future source
dates, unavailable side ratings or inconsistent identities preserve the
incumbent channel.

For an actually recomputed player lineup, confidence uses the resolved player
identities and each player's historical `games`/`last_ts`; the previous team's
roster date and continuity do not describe that new mean. Applied adjustments
without resolved identities fail closed. Duplicate players and future player
dates are rejected. Recent team sample counts should retain their declared
window (`sample_window_days`); they must not be described as lifetime support.
This window describes the source query ending at capture time; counts attached
to an older source may cover a shorter span before that source. The count's
certification timestamp must precede both the forecast and rating source.
Day-only source dates are conservative UTC-day proxies, not precise timestamps.

The corrected archive contains rating differences and an experience difference,
but lacks the eight certified per-side date/roster/availability/rating inputs
required for this policy. An experience difference cannot recover both sample
counts. `scripts/wpx_confidence.py` records that absence and saves the candidate
policy/source; it deliberately reports no historical accuracy score.

Integration, with the current forecast still unchanged:

```python
from lol_ticker import wpconfidence
state["prior_confidence"] = wpconfidence.capture(
    full_priors, as_of_ts=prediction_timestamp, teams=teams)
# An explicitly evaluated adapter may use this block's adjusted_priors.
# Preserve exact pelo_blue/pelo_red when a lineup mean is recomputed.
```

## Fourth early-game solo-queue prior

`lol_ticker/wpsqearly.py` uses the existing shipped `sqpairs.Scorer` as a new
isolated input. It distinguishes a valid zero score from missing coverage and
requires at least 80% of the 45 possible pair cells. Live metadata champions
are joined through role and participant ID, with missing, duplicate and
wrong-side identities rejected. A pinned scorer generation and its table hash
are cached; a nightly table replacement is detected at the next capture.
Live release versions such as `16.16.1` are normalized to `16.16` before
scoring, so same-patch tables cannot enter by tuple-ordering accident.

The score has its own fitted nonnegative, nonincreasing curve at minutes
0, 5, 10, 15 and 20. Its terminal coefficient is fixed to zero; every minute
at or after 20, and every uncovered draft, returns the baseline probability
exactly. Ridge 300, adjacent-knot penalty 70 and the 20-minute endpoint were
fixed before the screen. There is no constant late-game offset or core refit.

`scripts/wpx_sqearly.py` uses only the corrected, consumed archive through
September 2 and hash-verified chronological corrected-core caches. It does not
query PostgreSQL or the network. Since SQ coverage begins after patch 16.8,
the pre-May fitting blocks cannot train this channel. August is the
prespecified development block (training before July 4; baseline calibration
July 4–31); September 1–2 is a separate, very small later diagnostic. Both
models use the original earlier-block baseline Platt transform, preserving
the missing/late fallback exactly. Whole date/match clusters supply paired
intervals and games receive equal total weight.

| Block | Games | Covered training games | Baseline Brier | Candidate Brier | Change [95% interval] |
|---|---:|---:|---:|---:|---|
| August development | 999 | 1,456 | 0.14311519 | 0.14269089 | −0.00042430 [−0.00121583, +0.00043312] |
| September diagnostic | 19 | 2,102 | 0.14967899 | 0.15219011 | +0.00251112 [−0.00028199, +0.00596603] |

Development selection requires a negative upper paired Brier bound and no
log-loss regression. The interval includes zero, so **baseline is selected**.
The August raw-score slope is approximately 0.542 through minute 10, 0.518
at minute 15, and zero at minute 20. The interval on the early covered slice
also includes zero. September has only six clusters and cannot establish a
small improvement. Across the entire archive, 3,043 of 15,408 maps have
covered pair scores; coverage is 998 of 999 August maps.

This is a frozen-core residual screen, not a joint four-channel GAM fit or
new prospective promotion evidence. Pair cells come only from earlier patches;
the shipped scorer's global shrink constants are pinned from the SQ-only
table artifact and may include later scraped patches. That nuisance-parameter
limitation remains explicit. No pro outcomes enter the pair tables.

Integration for richer input capture:

```python
from lol_ticker import wpsqearly
state["sq_early"] = wpsqearly.live_capture(game_metadata)
# Experimental use only:
p = wpsqearly.predict(candidate, scores, covered, baseline_p, minute)
```

## Validation and artifacts

Twenty confidence/early-SQ/existing-pair unit tests pass. They cover future
metadata, missing factors, side reversal, rating-source identity, adjusted
lineups, valid zero SQ availability, decay constraints, exact late/missing
fallback and artifact roundtrip. The verified SQ run completes without the
small-matrix BLAS warnings seen in the initial run; explicit contractions
reproduce that run's numerical results. Models, predictions, source snapshots,
stage metadata and dependency versions are content-hashed in completion files.

Completed artifacts:

- `data/wpx/confidence_2026-10-04_final/`: fixed policy and metadata audit.
- `data/wpx/sqearly_2026-10-04_final/`: model artifacts, predictions,
  coverage, paired scores and reproduction metadata.

```sh
python3 scripts/wpx_confidence.py --out data/wpx/confidence_reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 scripts/wpx_sqearly.py --out data/wpx/sqearly_reproduction
python3 -m unittest tests.test_wpconfidence tests.test_wpsqearly tests.test_sqpairs
```

# Repaired models versus the market — September 4, 2026

The monthly-refitted rich GAM has lower historical Brier error than both
exchanges at reconstructed state times. Its advantage becomes inconclusive
when market prices are sampled 45 seconds later. At a 195-second offset,
approximating the observed upstream feed delay, both exchanges outperform
every reevaluated method. This supports prioritizing fresher state data; it
does not establish a live trading edge.

## Matched comparison

Lower Brier is better. Each game receives equal total weight. Within each
exchange, all columns below use exactly the same game states, including the
same model predictions. The exchanges cover different subsets. These scores
therefore differ from the earlier all-game rich-GAM score of 0.143160.

| Exchange | Games / states | Rich GAM | Market at state clock | Market +45 sec | Market +195 sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| Polymarket | 1,681 / 48,208 | 0.157977 | 0.165204 | 0.161493 | 0.148156 |
| Kalshi | 1,154 / 31,405 | 0.155546 | 0.161685 | 0.158000 | 0.144922 |

The paired differences below are **model minus market**: negative favors the
model. Intervals resample whole date/match clusters, keeping states and games
from a cluster together. They are exploratory, without adjustment for the
earlier search over model families.

| Exchange | Price lookup offset | Brier difference | 95% paired interval |
| --- | ---: | ---: | ---: |
| Polymarket | 0 sec | -0.007226 | [-0.011655, -0.002811] |
| Polymarket | 45 sec | -0.003515 | [-0.007958, +0.000996] |
| Polymarket | 195 sec | +0.009821 | [+0.005501, +0.014400] |
| Kalshi | 0 sec | -0.006139 | [-0.011463, -0.000813] |
| Kalshi | 45 sec | -0.002454 | [-0.007768, +0.002836] |
| Kalshi | 195 sec | +0.010624 | [+0.005475, +0.015766] |

Whole-date bootstrap sensitivity gives the same conclusions: the rich GAM
wins at zero offset, the 45-second intervals contain zero, and the market
wins at 195 seconds. Log loss has the same point-estimate ordering.

The repaired core GAM scores 0.159240 on the Polymarket cohort and 0.155255
on Kalshi. Rich features improve the Polymarket point estimate but slightly
worsen Kalshi's. None of the 14 families or two ensembles has lower Brier
than either exchange at the 195-second offset. The comparison reuses all
saved predictions; it does not fit or select a new market-specific model.

As a coverage sensitivity, using every available row separately at each
offset produces the same rich-GAM conclusion:

| Exchange | Offset | Games / states | Rich GAM | Market |
| --- | ---: | ---: | ---: | ---: |
| Polymarket | 0 sec | 1,681 / 52,953 | 0.147795 | 0.155282 |
| Polymarket | 45 sec | 1,681 / 52,129 | 0.149898 | 0.153105 |
| Polymarket | 195 sec | 1,681 / 48,421 | 0.157989 | 0.148157 |
| Kalshi | 0 sec | 1,156 / 35,273 | 0.145625 | 0.151911 |
| Kalshi | 45 sec | 1,156 / 34,674 | 0.148290 | 0.150215 |
| Kalshi | 195 sec | 1,156 / 32,244 | 0.156617 | 0.145546 |

These sensitivity rows have different state composition across offsets;
the first table holds composition fixed to isolate the timing comparison.

## Timing and price limitations

The diagnostic covers May 1–September 2, 2026, using the saved chronological
monthly-refit predictions from the
[alternative-method reevaluation](alternative-methods-2026-09-04.md).
Historical alignment was derived retrospectively, partly from market reactions
and terminal values. Consequently, the zero-offset result does not prove
that the model and market had identical information available live.

The new runner reads each market's own reconstructed start, end and pauses,
requires alignment quality at least 0.4 and at least ten price observations,
and looks up the last stored observation at or before each target time.
It rejects missing observations and observations older than 60 seconds;
there is no fallback to a future price. It excludes the opening time-zero
state and lookups after reconstructed game end. Prices are oriented to blue,
duplicate timestamps are averaged, and available markets for the same
platform/game/state are averaged.

At zero offset, median observation age is 8.5 seconds on Polymarket and
12.5 seconds on Kalshi. At a 195-second lookup offset, median effective
market lead is 186 and 183 seconds respectively. Stored observations mix
trades, history/candle prices and book midpoints at second-level timestamp
resolution. They are price proxies; this is a forecast-accuracy comparison,
with no executable spread, fee, fill or return calculation.

The underlying evaluation dates have already been consumed. These are
retrospective diagnostics, not fresh confirmation or grounds for promotion.

## Live evidence

The read-only ledger snapshot still evaluates the deployed v8 model, not
the staged v9 candidates or rich GAM. The older v11 protocol has only four
eligible resolved games per exchange under its 90-second maximum market
lead. Its paired intervals span substantial gains and losses.

The new v12 protocol has zero eligible resolved games on either exchange.
Its one resolved game with valid Kalshi quotes has a median upstream feed
age of about 194 seconds and processing latency below one second. The
delay is dominated by upstream state delivery. This sample cannot establish
an edge; the existing target remains 100 eligible games per exchange.

## Reproduction and validation

Run `python3 scripts/wpx_market_v9.py --output data/wpx/market_v9_rerun`.
The runner uses a read-only, repeatable-read database transaction and the
saved frozen and monthly-refit prediction arrays; it does not train models,
enroll live games, or alter the evaluation registry. The main tables above
use monthly refits. Frozen-model comparisons are retained in the JSON too.

Outputs are in `data/wpx/market_v9/`: `plan.json` records source, dataset,
prediction and protected-artifact hashes; `results.json` contains every
family and both bootstrap schemes; `paired_prices.npz` retains matched
prices and ages; and `live_ledgers.json` contains the ledger snapshot.
The run read 5,092 market alignments in 35.1 seconds. Production artifact
and registry hashes remained unchanged. The full suite passed 120 tests,
including future-price rejection, exact timestamps, stale observations,
invalid values and duplicate timestamps; `git diff --check` passed.

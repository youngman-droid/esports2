# LPL postdraft replay — September 12, 2026

LPL is now included in the historical postdraft betting analysis. The old
input builder required gameplay timelines, which none of the 325 LPL maps
in this evaluation window had. The new `lol_ticker/wppostdraft.py` exporter
uses completed drafts and pregame information without that requirement.

Evaluation window: May 1 through September 2, 2026, inclusive. The available
LPL data and selected LPL bets end August 30. Dollar sizes below are **principal
per bet, with fees paid additionally**, using continuous shares and holding to
settlement. These are historical replay estimates, not executed account P&L.

## Added LPL results

| Platform | Bets | Wins–losses | $10 net | $20 net | $100 net | ROI including fees |
|---|---:|---:|---:|---:|---:|---:|
| Kalshi | 170 | 101–69 | −$142.71 | −$285.42 | −$1,427.10 | -8.18% |
| Polymarket | 259 | 145–114 | −$197.18 | −$394.36 | −$1,971.82 | -7.49% |

At $10 per bet, LPL gross profit before fees was −$98.54 on Kalshi and
−$154.23 on Polymarket; fees add $44.17 and $42.96, respectively.

## Updated league breakdown, after fees

| League | Kalshi bets | Kalshi $10 | Kalshi $20 | Polymarket bets | Polymarket $10 | Polymarket $20 |
|---|---:|---:|---:|---:|---:|---:|
| LCK | 36 | +$11.95 | +$23.90 | 106 | +$55.34 | +$110.68 |
| LPL | 170 | −$142.71 | −$285.42 | 259 | −$197.18 | −$394.36 |
| LEC | 43 | +$9.32 | +$18.64 | 61 | −$46.48 | −$92.97 |
| LCS | 46 | +$191.81 | +$383.62 | 68 | +$160.80 | +$321.60 |
| LCP | 45 | −$34.46 | −$68.92 | 82 | −$30.82 | −$61.63 |
| CBLOL | 24 | +$60.74 | +$121.48 | 39 | +$93.12 | +$186.23 |
| MSI | 30 | −$223.88 | −$447.76 | 34 | −$196.80 | −$393.59 |
| Domestic majors total | 364 | +$96.65 | +$193.30 | 615 | +$34.77 | +$69.54 |
| Including MSI total | 394 | −$127.23 | −$254.46 | 649 | −$162.03 | −$324.05 |

## Model and coverage

- The recovered historical v8 model was fit on 12,311 games strictly before
  May 1, 2026. Its 99,408 saved held-out predictions match bit for bit, with
  maximum absolute difference zero. LPL evaluation outcomes are not used to
  fit the recovered model. Original v8 training-input and calibration
  limitations are retained; this does not claim a repaired model evaluation.
- Inputs are the ten completed champion picks plus the same pregame team/player
  strength, form and earlier series results used by the existing model. This
  is not a champion-only model. Every gameplay feature is neutral at time zero.
- The exporter accepts 324 of 325 LPL maps. Map 80200 has conflicting draft
  records and is rejected; it has no matching betting records.
- Of the accepted maps, 323 have full OE pregame priors. Map 81001 has a flagged
  neutral OE fallback and no matched betting markets, so it contributes no bets.
  All accepted champions are recognized by the frozen model vocabulary.
- Kalshi: 186 maps with paired markets; four lack fresh paired quotes and
  twelve have no positive model value, leaving 170 bets.
- Polymarket: 271 maps with paired markets; eleven lack fresh paired quotes and
  one has no positive model value, leaving 259 bets.
- All earlier major-league ledger rows are retained unchanged. Their existing
  exclusions, including contaminated time-zero health inputs, remain in force.
- LPL uses the current historical OE ratings snapshot. The old majors retain
  their archived input snapshot. An independent sequential reconstruction
  verified all available LPL OE and gol.gg priors before each outcome update,
  within float32 storage rounding (less than 0.00015 Elo). Backfilled OE inputs
  differ from the archived majors snapshot: across 856 reconstructable maps,
  neutral-state forecasts differ by 0.344 percentage points on average, with
  32 maps above one point. The fitted model is identical; input vintages differ.

## Price, selection and fee assumptions

Quotes use the same earlier replay cutoff: the latest stored quote at or before
OE game start plus 120 seconds. Both sides must be at most 60 seconds old. The
model receives no gameplay information, but prices at this cutoff may already
reflect early play. Historical completed picks do not prove when a live draft
lock was observed. Both outcome markets must belong to the same event/condition
and settle consistently with the game label.

For each map and platform, select the side with the greatest positive
`model_probability / entry_price - 1`, then deduct fees. There is no new fee-aware
selection filter. Kalshi uses native YES asks; Polymarket uses recorded token
prices because historical executable asks are unavailable. Consequently the
Polymarket P&L is a paper estimate. Neither replay models slippage, available
size, integer-contract constraints, financing or bankroll limits.

Fees use `shares × rate × price × (1 − price)`, where shares equal principal
stake divided by entry price. Kalshi uses the standard 0.07 coefficient; the
prior verification found no KXLOLMAP fee-schedule changes or event overrides.
Fees are rounded upward to cents before May 28, 2026 and to 0.0001 dollars from
that date, after the initial six-decimal upward rounding. Settlement fees are
zero. Account-specific historical waivers cannot be reconstructed. Sources:
[Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf),
[fee rounding](https://docs.kalshi.com/getting_started/fee_rounding), and
[changelog](https://docs.kalshi.com/changelog/index).

Polymarket sports taker fees use 0.03 before July 10, 2026 UTC and 0.05 from
that date, rounded to five decimals. BUY fees are paid as additional collateral.
Current market metadata is checked for enabled sports fees, but the applicable
historical rate follows the dated change rather than today's snapshot. LPL has
117 selected bets at 0.03 and 142 at 0.05. Sources:
[Polymarket predictions changelog](https://docs.polymarket.com/changelog/predictions)
and [trading fees](https://docs.polymarket.com/trading/fees).

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 research/wpx_recover_v8.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 research/wpx_postdraft_bets.py
python3 -m unittest tests.test_wppostdraft tests.test_wpx_recover_v8 \
  tests.test_wpx_postdraft_bets tests.test_update_pipeline tests.test_wpx_major
```

The 22 targeted tests pass. They cover draft completeness and side orientation,
feature independence from current/future outcomes, no gameplay table access,
recovery mismatch rejection, CLI guards, fresh quote and settlement checks,
fee schedule boundaries, and payout arithmetic.

Output directory: `data/wpx/postdraft_lpl_2026-09-12/`.

- `recovery.json` and `recovered_v8.npz`: verified model and provenance.
- `postdraft_inputs.npz` and its manifest: draft-complete inputs and exclusions.
- `predictions.json`, `quoted_markets.json`: frozen inference and quote evidence.
- `betting_plan.json`: source hashes and replay assumptions.
- `lpl_bet_ledger.json`, `combined_bet_ledger.json`: new and combined bets.
- `betting_results.json`, `league_results.json`: machine-readable summaries.

The export and replay use read-only database connections and isolated artifacts.
This change integrates LPL into historical postdraft analysis; it does not add
LPL live gameplay data or deploy a replacement model.

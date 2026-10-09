# Production combination — October 4, 2026

Activated **objectives + trajectories + composition + early SQ** at the user's explicit request: “Set those as the production weights.” This is a manual promotion. The historical statistical gate remains recorded as inconclusive; no evaluation registry was changed to claim a pass.

Production uses the exact August-tested joint artifact and its matching corrected core, pregame model, champion-state weights and shared Platt calibration. No weights were refitted. The packaged scorer reproduces all 32,230 August predictions exactly, with upstream/core agreement on 50,572 calibration and validation states. The evaluated SQ table was recovered from an existing local backup and copied into the immutable release.

The active pointer is [live_stack.json](../data/wpx/live_stack.json). Its hash-bound [bundle](../data/wpx/production_combination_2026-10-04/bundle.json) contains the base, joint weights, calibration, source provenance and supported features. The production model kind is `wpgam_v9_objective_trend_composition_sq_v1`.

Live scoring applies the raw core, joint residual and shared calibration once, in that order. It uses only supported remaining-buff timers, exact 120/300-second trajectory anchors, four sourced composition proxies and prior-patch SQ. Unsupported or missing inputs contribute zero. SQ is exactly zero from minute 20. No-champion comparisons also remove composition and SQ. A separate exact-history tracker preserves the previous research capture, and an explicit gameplay major/minor mapping joins live server builds to matching saved static catalogs.

The shadow recorder and dashboard were reloaded. Shadow now records the combination's bundle and stack hashes under protocol `shadow_v13_production_combination-e46078bca00a`; older ledgers remain intact. The existing historical-odds teacher is withheld for this model because it was calibrated for another base. The quote recorder continues running.

Validation: 406 tests run successfully, one database test skipped. Hash-tampering/fallback, calibration ordering, no-champion inference, unavailable inputs, clock consistency, SQ cutoff and source binding are covered. A live-input fixture enabled all four families, the production dispatch reported the new identity, and the local dashboard returned HTTP 200. These checks verify serving; no new live accuracy result is claimed.

The previous incumbent GAM and legacy model files are preserved. Invalid combination artifacts fall back to the incumbent with an explicit warning. Rollback restores [live_stack.previous.json](../data/wpx/production_combination_2026-10-04/live_stack.previous.json) to `data/wpx/live_stack.json` atomically; no model retraining is needed. Restart the scoring services after rollback to start a clean capture history.

See the [activation record](../data/wpx/production_combination_2026-10-04/activation.json), [deployment hashes](../data/wpx/production_combination_2026-10-04/deployment_completion.json), and [combination results](model-combinations-2026-10-04.md).

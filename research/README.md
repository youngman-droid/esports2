# research/

Experiment runners and one-off analyses: model-candidate fits and gates
(`wpx_*`), the TGV draft-model reconstruction (`tgv_*`), the solo-queue pair
studies (`sq_*`) and supporting modules. Nothing here runs in the recorder,
shadow recorder, dashboard or nightly update; `scripts/` keeps the operational
tools (update, daemons, backup, feed backfill, player pipeline, promotion).

Run from the repository root, e.g. `python3 research/wpx_combinations.py ...`;
modules import each other as `from research import ...`. Experiment manifests
under `data/wpx/` written before 2026-10-09 record these files at their old
`scripts/` (or `lol_ticker/`) paths; the recorded SHA-256 hashes still match.

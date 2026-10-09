# Rebuilding TGV's methodology with independent data

The earlier identification limit concerns recovering TGV's exact checkpoint from its published scores. It does not prevent rebuilding the methodology by measuring composition inputs from independent match data and fitting new coefficients. This is the actionable reconstruction path.

## Methodology to reproduce

The recovered draft score uses a logistic link, patch/role/champion strengths, symmetric within-team synergy, antisymmetric opposing-team matchups, professional familiarity, and composition. The source-described composition form is:

`C(B,R) = linear(B)-linear(R) + Q(R)*L(B)^2 - Q(B)*L(R)^2`

`L(T) = log(magicDpm(T)/physicalDpm(T)) - learnedCenter`

`Q(T) = balanceWeight + interactionWeight * normalizedTankiness(T)`.

The linear channels are true DPM, total DPM, tankiness, and gold demand. Team values are built from champion/role profiles. These descriptions were verified in the saved TGV review client. Normalization, smoothing, life-count convention, training penalties, and transfer-weight assignments remain implementation choices to estimate and validate, rather than facts recovered from TGV.

Professional familiarity has the recovered release-specific form `0.0789687760*n/(n+1.0362480792)`, with n being prior champion appearances in that role. Independent replication can test that fixed curve against a curve fitted only on the training period. The daily puzzle's artificial modifiers are a separate mechanism.

## Independent sources and their jobs

| Source | Useful coverage | Material limitation |
|---|---|---|
| [Riot Match-V5](https://developer.riotgames.com/apis#match-v5) | Primary collection interface for raw match details; use to assemble a contemporary solo-queue corpus | Requires a valid Riot API key and a collection window; a list of Challenger players does not imply all participants in their matches are Challenger |
| [GPTilt Challenger match dataset](https://huggingface.co/datasets/gptilt/lol-basic-matches-challenger-10k) | Public Riot-derived matches, participants and event tables; verified composition fields in downloaded sample | Sample inspected here covers 2025 patches. Dataset card specifies CC BY-NC 4.0. Suitable for this offline research; not assumed suitable for commercial deployment |
| [Oracle's Elixir data dictionary](https://lol.timsevenhuysen.com/matchdata/match-data-dictionary/) and our existing OE-derived history | Professional identities, roles, champions, dates, patches, results and historical familiarity; total damage and gold variables | The dictionary is historical; inspect each export's actual schema. Its earned-gold share differs from total-gold share. Do not assume it includes physical/magic split or self-mitigation |
| [GPTilt professional-match archive](https://huggingface.co/datasets/gptilt/lol-esports-matches) | Leaguepedia-derived role-ordered picks, bans, teams, results and tournament links | Its games table is not a participant-statistics or per-game player-roster table; it cannot alone provide player comfort |
| [Riot Data Dragon documentation](https://support-developer.riotgames.com/hc/en-us/articles/22698698001939-League-of-Legends) | Versioned champion IDs, names and static assets | Static champion stats cannot substitute for observed match damage, gold demand or self-mitigation |

The GPTilt professional archive also supplies entity-alias linkage and source URLs. Its card states CC BY-SA 3.0 and describes an overwrite-only snapshot. Pin a revision when importing. The discovered dataset revisions and public sample responses are saved under `data/tgv/independent-sources/`; the sample viewer response is retained verbatim, since its cached schema need not match the repository's current README.

## Measurements already verified

Downloaded the public Challenger sample's participant and match tables and joined them on match ID. Five complete games yield **50 usable player rows**, covering **patches 15.7 and 15.8**. The partial sixth game is excluded. Both teams have five unique roles. Derived total-gold shares sum to one per team. The three damage-type totals differ from the API total by at most two raw damage units in this sample; the normalizer records this discrepancy and defines its total as the sum of those channels.

The working normalizer is `research/tgv_external_sample.py`; output is `data/tgv/independent-sources/composition-sample.json`. It computes:

- Physical/magic/true damage to champions divided by game minutes.
- Total DPM as their sum.
- Gold demand candidate as player gold earned divided by team gold earned.
- Tankiness candidate as `(damageTaken+selfMitigated)/(minutes*(deaths+1))`.

The last two definitions are explicit candidates. TGV's exact gold convention and treatment of lives are not established. The output is historical measurement data, not a trained model or a claim of predictive accuracy. It excludes account identifiers not needed for composition.

## Training reconstruction

1. **Create a dated raw corpus.** Retain patch, region, queue, roles, champions, start time, duration, results and the verified damage/gold fields. Deduplicate match IDs and reject incomplete teams, invalid roles and remakes. Pin source revisions and retain row counts by patch and role.
2. **Estimate champion profiles from earlier games.** Fit positive damage/tankiness rates and gold-share estimates by champion, role and patch, shrinking sparse cells toward role and adjacent-patch baselines. Select shrinkage and pooling on chronological validation. These are new estimates from independent observations, not values inferred solely from TGV scores.
3. **Fit the solo-queue draft foundation.** Regularized logistic strength, synergy and matchup terms provide a large-sample starting point. Enforce the known side-swap symmetries. Keep champion-statistics estimation and outcome-fitting cutoffs explicit.
4. **Transfer to professional outcomes.** Fit composition coefficients, balance center, normalization and transfer scales with team/player strength as nuisance controls. TGV publishes a four-positive-scale/two-outcome-weight transfer label, but the exact assignments are unknown; any assignment we choose must be labeled an independent implementation.
5. **Add strictly historical comfort.** Use player/champion/role appearances before the forecast date. Our existing OE-derived snapshot already provides professional history for this work. Neither current-game outcomes nor current-game realized damage may enter that game's pre-draft predictors.
6. **Compare on common future blocks.** Evaluate controls-only, controls plus draft, and controls plus draft/comfort/composition on the same maps. Tune on validation and freeze the final test window. Compare log loss, Brier and calibration; isolate draft plus comfort by removing team/player nuisance contributions and side bias from the attribution. No historical accuracy claim should use a September checkpoint on earlier games as if it were available then.

This approach can reproduce the method and deliver an independently trained model. Exact equality to TGV's proprietary checkpoint is a separate question. The immediate data gap is contemporary raw composition coverage, not mathematical access to hidden factor values; the 2025 public sample already verifies that the required measurements are obtainable elsewhere.

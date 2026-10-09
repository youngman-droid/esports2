# Player impact rankings

Open `/players` in the local LoL Ticker dashboard. This experimental worldwide
professional-player board uses the existing Oracle's Elixir CSVs under
`data/oe/`. It includes every eligible player represented in those files;
coverage is not a census of all professional or solo-queue players.

```sh
python3 -m lol_ticker.player_ratings build --output data/player_ratings/ratings.json
python3 -m lol_ticker dashboard
# http://127.0.0.1:8090/players
```

The normal data update refreshes this artifact after Oracle ingestion. Failure
is nonfatal and preserves the previous artifact. Dashboard requests read the
cached JSON without fitting or opening the match database. The ranking model
does not replace production live win-probability priors.

## What the rating measures

The main rating is **wins added per 100 games**, fitted from match wins directly.
A recency-weighted logistic model has five positive player indicators for blue
and five negative indicators for red. Jointly estimated player and competition
coefficients adjust for teammates, opponents, champion-role selection and
blue-side context by league and patch. Individual effects shrink toward zero;
gold outcomes and lane-gold priors do not enter this model.

A player's total log-odds effect `b` includes individual and competition effects.
Displayed impact is `100 * (sigmoid(b) - 0.5)`: additional expected wins over 100
otherwise evenly matched games against a zero-effect reference player. For
example, +5 represents 55 expected wins instead of 50. This standardizes playing
time; it is not observed or cumulative career wins and is not a replacement-level
or causal estimate. Lineup contributions add in log odds; displayed rates do not
add. Results alone cannot separate teammates who always play together.

The role-relative index is 50 plus ten times the within-role standard score.
Career-by-age uses this historical index expressed in standard deviations;
raw wins added is available for each observation and current/peak ordering.
Intervals transform approximate conditional posterior log-odds intervals through
the same sigmoid, keeping endpoints within −50 to +50 wins per 100 games.

CLI rebuilds default to `--outcome wins`, including daily updates. The legacy gold
model remains available with `--outcome gold` or `build_ratings(outcome="gold")`
for reproducibility.

The approach is inspired by RAPM's joint lineup adjustment and DARKO's recency
and shrinkage ([DARKO methodology](https://www.darko.app/about)). It does not reproduce DARKO or implement its proprietary stat
projections or Kalman filter. Regularization and decay settings are fixed
exploratory choices, not established optimal settings for League of Legends.

## Reading the board

- Search by player, team or league; filter roles, leagues, games and activity.
- The scatter colors players by role and plots impact against available sample
  evidence. Select a player to inspect estimates and historical ratings.
- Monthly history uses only games available at that snapshot cutoff.
  Role-relative normalization also uses that snapshot rather than later data.
  The default retains the last 24 completed months plus older annual snapshots and the current fit.
- Raw game counts include older games; decayed games sum recency weights.
  Activity is defined relative to the artifact's cutoff.
- Intervals are conditional model uncertainty, not calibrated rank confidence;
  model misspecification, team systems and sparse international links remain.
- Disconnected competition groups cannot be ordered from match evidence alone.
  Cross-group comparisons rely on model centering and priors.
- Stable Oracle IDs remain distinct. Names without stable IDs use scoped
  fallback identities. An ID change can split history; duplicate names do not
  justify automatically joining people.

Published birthdates are matched through exact Oracle ID hashes of canonical
Leaguepedia pages and unambiguous published redirects. The age scatter and
cohort rankings use age at the artifact cutoff. Missing birthdates remain in
an explicit Unknown age cohort; dates are never estimated from debuts or names.
Career-peak mode uses each player's age at their sampled peak instead.

The local match archive now includes all official OE annual files for
2014–2026. Coverage differs by season (for example, 2014 lacks regular LCK
matches), so an early archive is not complete career coverage. Annual snapshots
extend trajectories through the older seasons; recent history remains monthly.
A career peak is the largest sampled wins-added estimate with at least 20
decayed games. Sampling and career length affect this statistic; it is not a
GOAT score or an age/era-adjusted causal measure. Peak selection uncertainty
is not shown as the current fit's interval.

`/players/archive` exposes 4,943 early match/series results and 2,025 individual
game records, including WCG 2010 and Season 1 Worlds. These lack individual
reliable complete player lineups and roles, so they cannot enter individual win fits.
2010 records carry an event window rather than invented game dates. No complete
2009 competitive archive has been found.

## Validation and provenance

The JSON includes source fingerprints, date range, coverage, parameters,
competition-group diagnostics and chronological backtests. The final 90 days
are scored using earlier training games against a logistic team-only model, side-only and coin-flip baselines. Brier score,
log loss and winner accuracy evaluate future game win probabilities,
not ground-truth individual player quality. Historical outcomes were already
available to the project; these are retrospective checks, not untouched
prospective validation.

```sh
python3 -m unittest discover -s tests -p 'test_player_r*.py'
```

Rebuild to incorporate updated CSVs. The artifact records its cutoff and source
dates and is atomically replaced only after a successful build.

## Competition adjustment

The joint model now fits a shared competition coefficient alongside individual
player deviations. A domestic match's five positive and five negative shared
coefficients cancel; cross-competition games identify relative strength.
The shared coefficient uses ridge penalty 2, while individual deviations retain
penalty 100 and a zero-effect prior. Log-odds player impact is their sum; the
display transforms this total into wins added per 100 games. Conditional
uncertainty includes shared-region variance and player/region covariance.

International team affiliations use the domestic team history known at game
time, with majority roster history as fallback. A single academy substitute
does not reclassify an entire international team. OGN/LCK, NA LCS/LCS and EU
LCS/LEC labels are normalized for this purpose. International events and cups
preserve domestic affiliations. Missing prior affiliations remain unlinked.
This prevents one international or cup appearance from inflating an academy
competition or erasing a major-league player's competition adjustment.

The shared coefficient is still uncertain: qualified teams are selected samples,
regional structures change, and a few bridges do not identify whole-league
quality perfectly. There are no player-specific boosts or rank constraints.
Chovy and Faker are sanity checks on the resulting estimates, not fitting inputs.
Current form and career peak deliberately answer different questions.

## Career observations by age and era

The default view follows the [age-band reference](https://x.com/tak_01_/status/2105519904371671228/)
by allowing the same source identity to appear at different ages. Each dot is
one player-season: the latest eligible fitted snapshot in that calendar year
with at least 20 decayed games. Recent seasons have monthly snapshots; earlier
seasons have annual snapshots. This is a sampled recency-weighted estimate,
not a fit to only the games in that season. Unknown birthdays remain unknown.
Coverage counts are recorded in the current artifact and reflected by the page.
Faker has separate sampled observations across his recorded career ages.

The vertical **era score** is `(historical role score - 50) / 10`. The underlying
role score was normalized against same-role peers at that historical fit, with
recency evidence weights capped at 100 games. Thus 1 means one standard
deviation above those peers. This standardizes relative standing across dates;
it does not establish equal absolute ability across eras or remove changes in
data coverage. Source ratings and age are never changed to achieve a desired
rank. Raw wins added / 100 games remains available in the table and selected-season profile.

Age bands use integer age at the snapshot, with deterministic horizontal jitter
for readability. Individual-age ranks are computed before player search.
Age-specific P90/P80/P70/median curves require at least 10 observations per age
and use the role/region/evidence/activity selection before search, so searching
for Faker does not redefine his comparison group. Selecting a season highlights
the visible observations of that same identity and connects its career points.
The spotlight uses the selected historical team, role, league and rating,
never its current metadata or current uncertainty interval. CSV exports retain
source identity, season, fitted date, age, rank group, raw impact and era score.

Checks: `node --test tests/player_age_seasons.test.js` tests dated identities,
snapshot selection, historical metadata, era scores, birthday boundaries,
unknown ages and insufficient evidence. Existing Python player/API/pipeline
tests also pass. Current-form and career-peak views remain available.

The comparison picker selects up to six stable player identities, including
retired players. Each selected player has a distinct color and a career line.
The comparison table can align historical observations by integer age or by
calendar season, keeping dates, teams and raw impact in each cell. Multiple
observations at the same age are retained, and missing observations are never
interpolated. Current-form and career-peak views also support side-by-side
comparison. Existing role, region, age, activity and evidence filters still
apply. Player search is ignored while comparing; reference percentiles and
ranks retain their wider comparison population. Players can be added through
the picker or selected-player profile, removed individually, or cleared together.
`node --test tests/player_comparison.test.js` checks alignment, missing data,
identity selection, repeated ages and the current/peak matrix.

Each solve drops observations older than eight half-lives (1,200 days by
default, weight below 1/256) to make full-archive rebuilds practical. Older
records stay in coverage, identity history and earlier historical fits.

## Data refresh and attribution

```sh
# Download missing official historical seasons; existing annual files are retained.
python3 scripts/player_archive.py
# Rejoin downloaded directory/birthday snapshots by exact wiki identity hash.
python3 scripts/player_identity_metadata.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -m lol_ticker.player_ratings build
```

The downloader validates the CSV schema and writes each year atomically. Failed
network requests retry; the existing daily 2024–2026 source files are preserved.
The normal update pipeline reuses the downloaded archive and sourced metadata.
CLI defaults detect `data/player_ratings/birthdays.json` and
`identity_aliases.json`; explicit `--birthdays` and `--aliases` remain available.
Birthdate provenance is in `birthday_provenance.json`. Exact canonical wiki-page hashes take precedence over recycled display-name aliases (for example, Ruler). The metadata join rejects
ambiguous hashed aliases and never joins solely on a displayed name.

Sources: [Oracle's Elixir official downloads](https://oracleselixir.com/tools/downloads),
[Leaguepedia birth-year directory](https://lol.fandom.com/wiki/Category:Player_Birth_Year_Categories),
[GPTilt esports identities](https://huggingface.co/datasets/gptilt/lol-esports-entities),
[GPTilt esports matches](https://huggingface.co/datasets/gptilt/lol-esports-matches),
and [Leaguepedia WCG 2010](https://lol.fandom.com/wiki/2010_World_Cyber_Games).
Leaguepedia-derived facts and metadata are licensed CC BY-SA 3.0; retain attribution
and compatible share-alike terms when redistributing them. Raw Parquet snapshots,
source cards, extracted birthday tables, early records and derived artifacts live
under `data/player_ratings/`. The isolated Parquet import requires `pyarrow`;
normal model fitting and serving do not. Birthdates are a sourced snapshot and
need a new directory extraction to pick up newly published dates.

The win rebuild accepts complete ten-player games even when gold is missing.
The artifact records current coverage and frozen 90-day holdout metrics. These
are predictive diagnostics, not ground-truth validation of individual ranks;
no player receives a name-based boost.

The October 4, 2026 cutoff includes 102,623 games and 12,621 identities. Its frozen
3,382-game holdout scores win RAPM at Brier 0.235677, log loss 0.663796 and
60.47% winner accuracy, versus the team baseline's 0.244951, 0.683013 and 55.44%.
Brier score and log loss are lower-is-better predictive checks. These fixed
exploratory fits have not been tuned or validated as causal player effects.


## Geographic regions

The player explorer filters and displays geographic regions rather than source
circuits. Domestic circuits and their academy divisions share a region. Regions
are Korea, China, EMEA, North America, Brazil, Latin America, Asia-Pacific,
South America (LTA South), and mixed Americas (LTA-wide observations). Asia-Pacific
includes Japan, Vietnam, Taiwan/SEA and Oceania. These are playing affiliations,
not nationality. Historical and peak views use the league at that snapshot.
Recognized international events fall back only to dated domestic history at or
before that fit; unmapped domestic circuits stay Unknown, and international-only
players stay International / home unknown. Source circuit is retained in region
cell tooltips and CSV exports. This presentation mapping does not change model
coefficients or merge the model's competition controls.

Sources: [Riot's EMEA regional rules](https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/1/18/ERL_2025_Rulebook.pdf/revision/latest?cb=20250104012728),
[Riot's Pacific ecosystem](https://lolesports.com/en-SG/news/lcp-2025-season-),
[Riot's Americas conferences](https://lolesports.com/en-US/news/introducing-league-of-legends-championship-of-the-americas),
[Leaguepedia's LCK Academy Series](https://lol.fandom.com/wiki/LCK_Academy_Series/2024_Season/1st_Championship),
and [Circuito Desafiante](https://lol.fandom.com/wiki/Circuito_Desafiante).
Checks: `node --test tests/player_regions.test.js` covers regional grouping,
academy aliases, historical transfers and unresolved affiliations.

# TGV Analytics compared with esports2

New evidence: [public-data reverse engineering](tgv-reverse-engineering-2026-09-16.md) subsequently recovered logistic-model provenance, complete public strength/interaction tables, and the count-based comfort curve. Earlier statements below that the model family and comfort estimator were unknown describe the initial page-only inspection, not the current findings.

Follow-up: the [isolated draft-plus-comfort study](draft-comfort-vs-tgv-2026-09-16.md) found an existing historical comfort experiment and tested its main proxies with our draft vocabulary. Comfort is absent from the deployed model, but is not a new idea for this repository. After controlling for overall player strength and experience, the incremental benefit was inconclusive. That study supersedes the comfort-priority recommendation below.

Inspected September 16, 2026. Sources: public homepage, rendered Champion Board and Rankings, and the public Daily Draft explanation/interface. This is a public-interface scrape and source-code comparison, not access to TGV's training code or a head-to-head prediction benchmark. No model was changed.

TGV has a more developed draft-decision product. esports2 has a substantially broader live-state prediction and market-evaluation pipeline. Public evidence does not establish which has better postdraft probabilities.

| Dimension | TGV public evidence | esports2 implementation |
|---|---|---|
| Main use | Draft evaluation and optimization | Pregame/postdraft and live win probabilities, market comparisons, event impacts |
| Champion effects | Role-specific strength, patch history, matchup and synergy scores | Live GAM has regularized signed champion effects and a separate champion-state channel; standalone draft models also include roles, patches, synergy and matchup pairs |
| Player specificity | Rankings explicitly use actual player–champion comfort | Team/player Elo, form, roster adjustments and earlier series results; the live GAM has no explicit player × champion comfort feature |
| Composition | Draft explanation includes tankiness, damage balance, total/true DPM and gold | Live gold and state inputs are extensive, but the production feature contract lacks an explicit draft composition representation of tankiness and damage mix |
| Decisions | Exact minimax is claimed for the constrained puzzle | Draft simulator scores a supplied action sequence; no corresponding minimax search in that implementation |
| Live state | No live-state probability interface found on inspected pages | Gold, momentum, CS, structures, objectives, buffs, deaths, HP, levels and game time |
| Evaluation | No numerical holdout, calibration or prospective benchmark found on inspected pages | Chronological studies, game-balanced Brier/log loss, paired intervals, immutable forward forecasts and frozen candidate gates |

TGV sources: [Champion Board](https://tgv-analytics.com/champions), [Rankings](https://tgv-analytics.com/teams), [Daily Draft](https://tgv-analytics.com/arcade). Local implementation: `lol_ticker/wpgam.py`, `lol_ticker/draft.py`, `lol_ticker/shadow.py`, `lol_ticker/wpcandidate.py`.

**What the scrape established**

The [Daily Draft explanation](https://tgv-analytics.com/arcade) describes a scoring model combining side, champion strength, player comfort, relationships and composition. Its solver considers optimal responses under limited champion pools, including flex and denial. Puzzle comfort is generated; it should not be mistaken for a measurement of a professional player's proficiency. Optimality is conditional on its scoring function and permitted actions, and does not establish real-world prediction accuracy. The training family, loss, regularization and calibration were not disclosed in the pages inspected.

The rendered [Rankings](https://tgv-analytics.com/teams) page said it used games through September 14, 2026 and displayed the top 100 active teams. Its draft column is a median of recent modeled draft advantages using original patches and actual player–champion comfort, with side bias removed and a neutral 50% baseline. It is not the probability of winning the next match against a named opponent. Examples observed:

| Team | Elo shown | Draft advantage | Neutral-baseline change | Coverage |
|---|---:|---:|---:|---:|
| Gen.G | 3265 | +0.029 | +0.7 percentage points | 10/10 |
| G2 Esports | 3089 | +0.185 | +4.6 percentage points | 10/10 |
| KT Rolster | 3087 | −0.278 | −6.9 percentage points | 10/10 |

These score/probability pairs are consistent with a logistic conversion: sigmoid(0.185) is approximately 54.6%. That is an inference about the display, not proof of a logistic-regression training procedure. The page also says its Elo history is reconstructed with the current model's champion ratings. Consequently, that retrospective chart is not an archived prospective forecast ledger.

The rendered [Champion Board](https://tgv-analytics.com/champions) exposed patches 16.12–16.18. On patch 16.17, top role, with the professional pick-rate filter set to zero, it showed 173 champions. Kennen displayed strength +130, matchup extremes +185/−256 and synergy extremes +027/−025, plus patch history. These are interface score units; their conversion to log odds was not verified. Scores for rarely or never professionally picked role combinations do not by themselves establish reliable estimates. The underlying data population and uncertainty treatment remain unknown.

**The important distinctions inside our own project**

We have three relevant estimators, and treating them as one would obscure the comparison:

1. The market draft model in `draft.py` predicts changes in market log odds. It is ridge-regularized, with pick/ban, role, patch and pair features. The README reports chronological R² around 0.02: it explains little of market repricing. This is a different target from predicting winners.
2. The standalone draft outcome model predicts winners using Elo plus draft features. The README reports log loss around 0.628 versus 0.634 for Elo alone. Code inspection shows its holdout is random, and feature-support selection occurs before splitting. These historical numbers are exploratory, not strong evidence of patch-forward generalization.
3. The serving win-probability stack uses the v8 constrained GAM alone (`data/wpx/live_stack.json`: weight 1.0; blend rejected). Pregame team/player priors and champion effects feed smooth time-dependent state effects. Its production champion representation is simpler than our separate draft model's interaction vocabulary.

The corrected-input challenger is separate from deployed v8. The [September 12 report](model-improvements-2026-09-12.md) documents 15,408 consumed games, HP timing repairs, inventory-channel ablation, a frozen candidate and a prospective promotion rule. Historical core monthly Brier was 0.143550 on 3,097 games; it is an in-game metric and cannot be compared directly with draft-only scores. Richer champion-resource families did not satisfy the study's selection rule. More features are therefore hypotheses to test, not evidence of improvement.

Our infrastructure provides more inspectable evaluation evidence, but that does not prove superiority over TGV. Our own reports identify historical input defects, delayed live feeds, and unresolved prospective accuracy questions. The [LPL postdraft replay](postdraft-lpl-2026-09-12.md) also shows negative historical returns in that cohort; draft modeling capability alone does not establish a market edge.

**What to adopt and how to compare fairly**

The most useful candidate addition is a heavily shrunk player × champion × role familiarity/comfort feature, built strictly from earlier games and tested beyond player Elo and champion strength. Next, test a small set of composition descriptors such as frontline and physical/magic damage balance, with patch-aware definitions. These are concrete gaps relative to TGV's exposed scoring components.

Before adding a search engine, establish that these features improve completed-draft predictions. Search can amplify errors by selecting the composition that exploits the evaluator most. For coaching or draft simulation, limited-pool search with legal role assignments and opponent replies is a worthwhile separate product enhancement.

A fair predictive comparison would freeze both model versions and collect probabilities for the same future professional maps at draft lock, with identical team/roster information and no gameplay inputs. Compare Brier, log loss and calibration using paired series/date clusters, and retain league/patch breakdowns. Test both draft-only effects at neutral team strength and full postdraft predictions if TGV exposes both. Current rankings and reconstructed history cannot substitute for those forecasts. Without TGV's callable scorer or export of appropriately timestamped forecasts, a numerical accuracy ranking is unavailable.

Recommendation: borrow the comfort/composition ideas and draft explanations; retain our forward validation discipline. Do not replace the deployed model or import TGV's displayed coefficients on the strength of this public comparison.

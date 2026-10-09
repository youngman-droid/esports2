# Mathematical recovery from fixed public observations

This pass performs no network requests. It uses the stored coefficient tensors, checksum drafts, challenge action sequence, legal pools, artificial comfort values, and minimax choice scores.

## New terminal composition scores

At action 19, Red makes the last pick. For each of the three published alternatives, the preceding 19 actions fix the entire draft. Enumerating all role permutations subject to the published team pools produces exactly one legal assignment for each side. Therefore the only unknown term in the full score is composition:

`composition = published blueValue - sideBias - unary - comfort - matchup - synergy`.

| Red final pick | Inferred composition |
|---|---:|
| Vladimir (8) | -0.01184868413685125 |
| Taliyah (163) | +0.019494131602568465 |
| Aurora (804) | +0.10061645219527965 |

The Vladimir value agrees with the separately published review composition to 1.12e-8, consistent with the engine/review numerical difference. Taliyah and Aurora are new inferred composition values, not copied breakdown fields. They are conditional on the verified additive factor formula and reported engine precision.

## Minimax constraints

At action 18, Blue has three published alternatives. Each leaves three legal terminal Red responses; all role assignments are unique. If V is the published move value, and K_ij is the recovered non-composition score for leaf i,j, minimization by Red gives:

`C_ij >= V_i - K_ij` for every j, with equality for at least one response.

This yields nine inequalities and three equality-disjunction constraints on a 3-by-3 terminal composition table. The known row from action 19 passes this check exactly up to numerical precision. No choice of the unknown minimizing response is silently assumed.

Reproduction: `PYTHONPATH=. python3 scripts/tgv_terminal_constraints.py`. Numeric results and full role-ordered drafts are in `data/tgv/20260916/terminal-composition-constraints.json`.

## Low-dimensional checksum hypotheses

Tested constant offsets, global component adjustments, and regularized individual role-pair adjustments with leave-one-patch-out diagnostics over 56 checksum drafts. The known-factor baseline has RMSE 0.09425 logits. Global component fitting gives 0.09895; role-pair shrinkage gives 0.09496–0.22603 depending on penalty. These fits do not explain the missing function or improve reliably over the baseline.

A minimum-norm additive correction can interpolate all 56 checksums to 2.78e-16 maximum error, but its design matrix has rank 56 for 6,055 unknowns. This is an explicit example of why exact interpolation alone is not recovery. The saved correction is diagnostic and has not been installed in any predictor.

Reproduction: `PYTHONPATH=. python3 scripts/tgv_math_constraints.py`. Results: `data/tgv/20260916/math-constraint-diagnostics.json`.

Next mathematical step: extend terminal enumeration to earlier minimax states and use their extremum constraints to constrain a structured composition function. Any structural assumptions must be tested against observations not used to fit them.

## Extending four moves backward: 180 terminal drafts

Enumerated all 180 complete drafts reachable after the canonical first 16 actions: four blue mids, three blue bots, five red tops, and three red bots. Every complete draft has exactly one legal role assignment under the published pools. Used 21 observed alternative-move values from actions 16–19.

An inverse-minimax mixed-integer program tests composition hypotheses by fitting terminal scores and enforcing the actual alternating max/min operations. It independently replays the resulting terminal table to check the solver's reported observation error.

| Composition hypothesis | Coefficient bound | Best max error | Proven lower bound |
|---|---:|---:|---:|
| Independent champion effects | 1 | 0.05135215 | 0.05135215 |
| Independent champion effects | 4 | 0.05193373 | 0 (time limit) |
| Arbitrary blue-team effect minus arbitrary red-team effect | 1 | 0.04249998 | 0.04249998 |

The bound-4 run did not prove optimality and is not evidence of an unbounded impossibility. The globally solved bound-1 cases reject only their explicitly bounded hypotheses. These are numerical MILP certificates with ordinary solver tolerances, not symbolic proofs. Files: `inverse-minimax-additive-bound1.json`, `inverse-minimax-additive-bound4.json`, `inverse-minimax-team-bound1.json`. Reproduction: `PYTHONPATH=. python3 scripts/tgv_inverse_minimax.py --structure champion|team --bound 1 --seconds 45` (new filenames use champion instead of additive).

## Bound-free rejection of independent champion effects

A separate algebraic identity avoids coefficient bounds and optimization. Fix red top Ambessa, blue top Olaf, jungle Zaahen, and support Milio. Let K(m,b,r) be the known-factor score for blue mid m, blue bot b, and red bot r. Terminal action 19 already determines the composition C(r) for canonical blue mid Annie and bot Xayah.

Under the additive hypothesis, all these terminal scores must be

`K(m,b,r) + C(r) + alpha(m) + beta(b)`.

Define `Q(m,b) = min_r[K(m,b,r)+C(r)]`. Action 18 then identifies every beta(b), since alpha(Annie)=0. The action-17 mid-pick values identify every alpha(m) through a maximum over b. The action-17 bot-pick values are separate checks: they must equal `max_m[Q(m,b)+alpha(m)+beta(b)]`.

| Bot choice | Predicted by additive identity | Published | Error |
|---|---:|---:|---:|
| Ezreal | -0.02068335424 | -0.01371140778 | -0.00697194646 |
| Xayah | 0.11883138865 | 0.11883138865 | 0 |
| Aphelios | -0.02871875869 | -0.03660392389 | +0.00788516520 |

The discrepancies are much larger than the engine/review numeric difference. This rejects the independent-mid-plus-bot effect hypothesis invariant to red response, without assumptions about coefficient magnitude. It does not separate within-team nonlinear effects from opposing-team coupling; those remain candidate explanations.

Reproduction: `PYTHONPATH=. python3 scripts/tgv_additivity_identity.py`; data: `additivity-identity-check.json`. Next: introduce structured interactions and use the same extremum constraints to determine which additional terms the data requires.

## Structured opposing-pick interaction fit

Extended the team-separable hypothesis by an interaction table between Blue's bot champion and Red's top champion. The table has eight gauge-fixed interaction coefficients. Together with the team effects and intercept there are 34 composition coefficients over the 180-draft subgame.

This structure fits all 21 observed late-move values with maximum independently replayed error 9.49e-15. This demonstrates a compatible local composition function, not that TGV uses this particular table. Six coefficients reach the imposed magnitude bound of one, and unobserved composition estimates range from -1.595 to +1.200, so exact interpolation should not be interpreted as a stable parameter recovery.

An attempted bot-versus-bot interaction fit hit its 40-second limit with a zero lower bound; it does not reject that alternative hypothesis.

### Prediction checks

Withheld action-17 Ezreal and Aphelios values and fitted the other 19:

| Fit | Training max error | Withheld max error |
|---|---:|---:|
| Unregularized feasible fit | 1.18e-14 | 0.13312921 |
| Minimum coefficient L1 norm, training tolerance 1e-7 | 9.99e-16 | 0.02589592 |

The minimum-L1 solution was globally solved within the stated bounded hypothesis, with coefficient norm 0.55108508. Regularization helps these two withheld values, but residual error remains far above numerical precision. These are exploratory checks on one puzzle, not an independent dataset or proof of TGV's parameterization.

Reproduction: `PYTHONPATH=. python3 scripts/tgv_inverse_minimax.py --structure team --cross bottop --holdout 17:81,17:523 --regularize --seconds 45`. Files use the `inverse-minimax-team-bound1` prefix with cross, holdout, and regularized suffixes.

### Explicit non-uniqueness witness

Starting with the exact 21-observation fit, subtracting 0.25 from the blue-team composition effect for any one of these mid/bot pairs changes 15 terminal predictions but leaves all 21 observed move values exactly unchanged:

- Galio / Aphelios
- Anivia / Ezreal
- Anivia / Aphelios

This is a concrete unobserved direction in the model, not only a parameter-count argument. Shifted coefficients are not asserted to remain inside the fitting bound. The same team-plus-cross-interaction function class still applies. Saved witness: `inverse-minimax-nonuniqueness-witness.json`.

Next mathematical step: incorporate earlier ban/pick observations. Removing alternatives changes which terminal states determine the extremum and may constrain these currently invisible team effects.

## Earlier bans: 320 drafts and 40 observations

Extended the mathematical game tree to actions 14–19. This introduces Varus as a blue bot candidate and Ziggs as a red bot candidate. There are now 320 complete role-unique drafts, 40 published alternative-move values, and 1,331 distinct min/max expressions. Ban passes are represented explicitly; changing a ban changes the legal later pick sets.

The prior blue-bot/red-top interaction family (47 parameters on this larger domain) reached a best maximum error of 0.00487423 across 64 deterministic starting points. This is a local-search result, not a proof of impossibility.

Adding the other opposing-role pair tables—blue mid/red top, blue mid/red bot, and blue bot/red bot—gives a 77-parameter composition function. One of 16 starts reproduces all 40 observed values with maximum error 4.87e-13. This provides a concrete compatible local function over the 320-draft subgame. The observations use only 22 active terminal states in that solution; many other terminal predictions remain unverified.

The fitted form is:

`C(m,b,t,r) = constant + B(m,b) - R(t,r) + I_mt(m,t) + I_mr(m,r) + I_bt(b,t) + I_br(b,r)`.

Here m,b are Blue's mid/bot and t,r are Red's top/bot. This is a surrogate parameterization inferred to match observed values; it is not a claim that these are TGV's original composition channels or coefficients.

### Independent ordered replay

A separate evaluator applies bans and picks in their original order and recursively computes min/max values from the fitted terminal table. It does not reuse the fitting graph. Across 33,004 memoized states and all 40 observations, it agrees with the graph to 5.55e-17. This checks the move ordering and legal-choice handling rather than only the fitted algebra.

Reproduction:

- `PYTHONPATH=. python3 scripts/tgv_ban_inverse.py --starts 64 --cross bottop`
- `PYTHONPATH=. python3 scripts/tgv_ban_inverse.py --starts 16 --cross all`
- `python3 scripts/tgv_replay_bans.py data/tgv/20260916/ban-inverse-all.json`

Results are saved in `ban-inverse-bottop.json`, `ban-inverse-all.json`, and `ban-inverse-all-ordered-replay.json`. All processes completed. Next: find a smaller compatible parameterization and validate against withheld move values, keeping interpolation distinct from generalization.

## Complexity reduction and a stricter prediction check

Added a linear-programming refinement that fixes the current min/max selections and minimizes the L1 norm of the composition coefficients while preserving the observed scores within 1e-8. This is a convex optimization within one strategy region, not a global minimum across all possible minimax strategies.

For the full 40-value fit, the coefficient norm decreases from 6.00360 to 3.97047 and the number of nonzero coefficients decreases from 77 to 63. Maximum replay error is 1.01e-8; independent ordered replay differs from the graph by only 2.78e-17. Saved as `ban-inverse-all-polished.json`.

The first 16-start withheld run did not fully fit its training values, so its prediction error was not used to judge generalization. A separate 32-start run, with a new random seed and an initializer fitted only to the permitted training observations, successfully matches all 38 training values within 1.01e-8. The initializer comes from the earlier regularized model that also withheld action-17 Ezreal and Aphelios; it does not use their targets.

The two withheld predictions remain poor:

| Choice | Published | Predicted | Error |
|---|---:|---:|---:|
| Ezreal at action 17 | -0.01371141 | +0.10829968 | +0.12201108 |
| Aphelios at action 17 | -0.03660392 | +0.10829968 | +0.14490360 |

These results show that exact fitting and local L1 refinement of the flexible interaction tables are insufficient for stable point prediction. They do not prove that every possible solution or regularization within that family must fail. The next mathematical hypothesis should constrain the interaction structure itself, rather than merely increase table flexibility.

Reproduction:

- `PYTHONPATH=. python3 scripts/tgv_ban_inverse.py --cross all --polish --polish-from data/tgv/20260916/ban-inverse-all.json`
- `PYTHONPATH=. python3 scripts/tgv_ban_inverse.py --cross all --starts 32 --seed 2516 --holdout 17:81,17:523 --polish --initial-from data/tgv/20260916/ban-training-only-initializer.json`

The training-only initializer and its provenance are saved. Both fits passed the separate ordered replay check. Next: test a low-rank opposing-team interaction motivated by damage balance multiplied by opposing tankiness; keep that structural hypothesis distinct from verified source coefficients.

## Rank-one opposing-team composition model

Tested `C(B,R)=F(B)-G(R)+sum_k U_k(B)V_k(R)` against the same 40 values. The rank-one and rank-two local searches initially stopped at the same 0.00151233 error: their active strategies incorrectly made passing a ban and banning Annie select the same terminal draft despite different published values.

Deterministic positive/negative perturbations of individual blue-team effects escaped that strategy region. A rank-one model then fits all 40 observations to 2.78e-17. Independent ordered replay agrees exactly across all observed states. The double-centered 16-by-20 composition matrix has leading singular value 1.09672; the next is 3.27e-16, confirming a single numerical interaction factor.

The rank-one parameterization has 72 stored parameters but 68 generic effective degrees of freedom after accounting for its offset, centering, and scaling freedoms. Its factors are arbitrary latent coordinates: neither has been identified as physical damage balance or tankiness. A rank-two alternative is structurally different, not automatically fewer parameters than the prior tables.

A canonical centered/scaled version of the exact rank-one fit is saved in `rank1-canonical-local-model.json`. It preserves every terminal composition to 1.11e-16. The file includes team ordering and the explicit evaluation formula.

### Withheld values remain unidentified

A separate rank-one fit used an initializer that excluded the two withheld action-17 values, then fitted only the remaining 38. It achieves training error 2.78e-17 but withheld maximum error remains 0.14490359. Its ordered replay also agrees exactly. Thus low interaction rank alone does not resolve the unseen move values; exact local compatibility is not yet predictive recovery.

Reproduction:

- `PYTHONPATH=. python3 scripts/tgv_ban_inverse.py --cross all --export-design`
- `python3 scripts/tgv_lowrank_inverse.py --rank 1 --starts 12 --initial-from data/tgv/20260916/ban-inverse-all-polished.json`
- `python3 scripts/tgv_lowrank_inverse.py --rank 1 --starts 33 --refine-from data/tgv/20260916/lowrank-inverse-rank1.json`
- `python3 scripts/tgv_lowrank_inverse.py --rank 1 --starts 12 --holdout 17:81,17:523 --initial-from data/tgv/20260916/ban-inverse-all-holdout17_81-17_523-polished-seed2516.json`

Analytic derivatives were checked by centered finite differences, with maximum error below 3e-11 at generic points away from strategy ties. No network access was used.

Next: constrain the latent team properties to sums of champion contributions where physically justified, then test the same withheld values. This targets structure rather than access or additional unconstrained tables.

## Compact additive-plus-bilinear reconstruction

Constraining both rank-one latent team factors to sums of champion contributions still fits all 40 observations to 1.39e-17, with 51 stored parameters when the remaining team base effects are unrestricted. Its withheld maximum error remains 0.14490.

Also constrained the base effects themselves to sums of champion contributions. After deterministic strategy-changing perturbations, this yields a 30-stored-parameter model that fits all 40 values with maximum error 1.55e-14. The separate ordered replay agrees exactly. Removing offset and scaling freedoms gives the compact expression:

`C = intercept + a·x + b·y + (u·x)(v·y)`.

Here x has six indicators for blue mid/bot substitutions from the reference picks, and y has seven for red top/bot substitutions. Normalizing u to unit norm and fixing its sign gives **26 effective degrees of freedom**. All coefficients, indicator ordering, fixed picks, and domain limits are saved in `compact-bilinear-local-model.json`. Reparameterization changes no terminal composition by more than 5.56e-17.

This is a concrete compact mathematical reconstruction on the observed subgame, not a claim that TGV originally trained these coefficients or that the formula applies to unobserved champions or patches.

### Prediction improvement

The compact model was independently fitted with action-17 Ezreal and Aphelios excluded, using only initializers that also excluded those targets. Strategy-changing refinement reduces its training maximum error to 1.77e-13. Its withheld predictions are:

| Choice | Published | Predicted | Error |
|---|---:|---:|---:|
| Ezreal | -0.01371141 | -0.03305642 | -0.01934501 |
| Aphelios | -0.03660392 | -0.00007861 | +0.03652531 |

Maximum withheld error improves from approximately 0.145 in the flexible rank-one/table models to 0.03653 here. This is still not exact recovery. These two values have been reused for exploratory model comparisons, so a fresh validation split is required before treating this improvement as general predictive evidence. The earlier smaller-subgame minimum-L1 experiment reached 0.02590 on the same two values; the current result is not claimed to dominate every earlier experiment.

Reproduction uses `scripts/tgv_lowrank_inverse.py` with `--factor-structure both --base-structure additive`. The exact fit and withheld fit use `--refine-from`, `--jump-all --jump-size .15 --starts 61`. Saved outputs have the suffix `refined-both-additivebase-alljumps`; the withheld version additionally includes `holdout17_81-17_523`. Both passed independent ordered replay over 33,004 states.

## Fresh validation and local identification

With the compact family fixed, a fresh split withheld action/champion pairs 14:1, 14:34, 15:67, and 16:82. Sixteen random starts used seed 4811 and no earlier fitted initializer. Selection used training error only. The best fit matches the 36 training observations to 2.67e-13, but its withheld predictions are:

| Action/champion | Published | Predicted | Error |
|---|---:|---:|---:|
| 14 / Annie | 0.21031050 | 0.21207489 | +0.00176439 |
| 14 / Anivia | 0.19733095 | 0.16785538 | -0.02947557 |
| 15 / Vayne | -0.01352171 | -0.01669917 | -0.00317746 |
| 16 / Mordekaiser | 0.59494841 | 2.00017524 | +1.40522683 |

Independent ordered replay agrees exactly over 33,004 states. This failure is therefore not a discrepancy between the fitting graph and ordered draft evaluation. It demonstrates that this fitting procedure does not reliably recover withheld values; it does not rule out every regularized solution in the family. These are newly withheld targets for this experiment, not previously unseen source data.

At the earlier exact 40-observation compact fit, the Jacobian of the selected terminal scores has rank 22 (threshold 1e-9). The model has 30 stored parameters and 26 effective degrees of freedom after four gauge freedoms. Thus at least four nongauge directions are unidentified to first order in this selected-strategy region. This is a local differential result, not by itself a proof of global nonuniqueness. Ties and strategy changes can affect identification. Singular values and caveats are saved in `compact-local-identifiability.json`.

Reproduce the fresh fit with `python3 scripts/tgv_lowrank_inverse.py --rank 1 --starts 16 --factor-structure both --base-structure additive --holdout 14:1,14:34,15:67,16:82 --seed 4811`. The result is saved as `lowrank-inverse-rank1-holdout14_1-14_34-15_67-16_82-both-additivebase.json`. The compact reconstruction remains a compatible local surrogate, not a recovered original composition model.

## Earlier bans add 22 observations and contested picks

Extended the inverse problem to actions 12–19, preserving the fixed earlier picks. This restores Ashe to Blue's bot pool and Viktor to both Blue's mid and Red's bot pools. There are 625 Cartesian role combinations, of which 25 illegally pick Viktor on both teams. The resulting domain has **600 legal completed drafts and 62 observed move values**.

`scripts/tgv_earlier_design.py` constructs the ordered draft graph using global champion availability. A Viktor ban removes him from both pools; a pick by either team makes him unavailable to the other. All 600 leaves have exactly one legal role assignment per side. The graph contains 41,955 distinct min/max nodes, representing 1,295,977 cached ordered states. The legal choices at each observed prefix exactly match the saved challenge's choices. The 320 previously studied leaves retain their known-factor scores to numerical precision, and the expanded graph reproduces all 40 previous predictions exactly when restricted to their canonical prefixes.

The graph and known-factor vectors are saved in `earlier-design-graph.json` and `earlier-design.npz`. `scripts/tgv_minimax_graph.py` evaluates the DAG in depth batches; independent serial evaluation agrees exactly on both random and tied terminal values. This speeds up inverse fitting without changing the min/max values. At ties, any returned active leaf has the exact extremal value.

### Expanded structural fits

| Composition hypothesis | Stored parameters | Best maximum error on all 62 values |
|---|---:|---:|
| Additive bases and one product of additive team factors | 36 | 0.00626775 logits |
| Additive bases and two products of additive team factors | 54 | 0.00159549 logits |

The one-product run used 16 random starts (seed 5811), followed by 73 deterministic coordinate perturbations of magnitude 0.15. The two-product run first used 16 random starts (seed 5812), then 109 perturbations from a rank-one initializer augmented with a small second factor (seed 5813). The latter initializer and its provenance are saved in `earlier-rank2-initializer.json`. All observations were training constraints in these expanded fits; neither result is a withheld prediction test.

Two opposing-team products are motivated by evaluating each team's damage mix against the other team's tankiness. However, the inferred latent factors are not identified as those physical quantities, and additive champion factors need not reproduce nonlinear damage-balance normalization. These are tested structural hypotheses, not recovered source formulas.

Neither optimization is globally certified. The best two-product fit still selects the same terminal draft for action-12 bans with different published values, and similarly conflates some action-14 values. Thus a local strategy-selection obstacle remains; the residual is not a proof that the hypothesis is impossible. Separate ordered replay over 1,295,977 states agrees exactly with its graph predictions.

Reproduction:

- `PYTHONPATH=. python3 scripts/tgv_earlier_design.py`
- `python3 scripts/tgv_lowrank_inverse.py --design-prefix earlier --rank 1 --starts 16 --factor-structure both --base-structure additive --seed 5811`
- `python3 scripts/tgv_lowrank_inverse.py --design-prefix earlier --rank 1 --starts 73 --factor-structure both --base-structure additive --refine-from data/tgv/20260916/lowrank-inverse-rank1-earlier-both-additivebase.json --jump-all --jump-size .15`
- `python3 scripts/tgv_lowrank_inverse.py --design-prefix earlier --rank 2 --starts 109 --factor-structure both --base-structure additive --refine-from data/tgv/20260916/earlier-rank2-initializer.json --jump-all --jump-size .15`

The expanded constraints supersede treating an exact fit to the smaller 40-value subgame as sufficient evidence. The unresolved objective is still the actual composition function, including its behavior beyond these observed subgames.

## A global ambiguity in the observed min/max values

There is an exact mathematical ambiguity that does not depend on Jacobian rank or a local optimizer. For any strictly increasing scalar function h and finite collection of values z:

`h(min(z)) = min(h(z))`, and `h(max(z)) = max(h(z))`.

By induction, h commutes with the entire pure min/max draft graph. Therefore, if h fixes every observed root value, transforming every terminal score by h leaves every observed root value unchanged. The 62 observations contain only 26 distinct target scores. Infinitely many increasing functions fix those 26 values while changing scores between them.

An explicit construction uses adjacent anchor values a<b:

`h(z) = z + 0.2*(b-a)*sin(pi*(z-a)/(b-a))` for a<z<b, and h(z)=z otherwise.

The derivative inside the interval is at least `1-0.2*pi = 0.37168`, so h is strictly increasing. This statement applies to any exact compatible terminal table, including the unknown true one, provided it contains terminal values inside the chosen interval. It does not require finding that table first.

For a numerical witness, `scripts/tgv_monotone_witness.py` additionally anchors every prediction of the current approximate two-product fit. It chooses the nonempty gap from 0.2439783514 to 0.5949484110. The transformation changes **153 of 600 terminal scores**, by up to **0.07019108 logits**, while preserving all 62 graph predictions exactly. The original and transformed maximum observation residuals are both 0.00159549; the transformation does not repair the approximate fit. Separate ordered replay of the transformed table agrees over 1,295,977 states.

Known unary, comfort, matchup, synergy, and side contributions K are retained: the alternative composition table is `C'=h(K+C)-K`. This is a statement about unrestricted composition tables on the fixed-roster subgame. The transformed C' is not established to belong to TGV's physical channel family or even the fitted bilinear family. The theorem also does not automatically apply to mixed role-assignment equilibria, which can involve averages rather than pure extrema; every terminal in this subgame has unique role assignments.

Consequently, recovering the original unobserved terminal scores requires independently justified restrictions on the composition function. Exact interpolation of these min/max observations alone cannot identify an arbitrary composition table. This explains why increasing fit accuracy is necessary for a compatible reconstruction but insufficient to establish the true model.

Reproduce with `python3 scripts/tgv_monotone_witness.py data/tgv/20260916/lowrank-inverse-rank2-earlier-refined-both-additivebase-alljumps.json`. The saved output has suffix `-monotone-witness.json` and contains the transformed terminal table, unchanged observations, formula, and limitations.

## Smooth optimization check

Tested whether replacing hard extrema temporarily with log-sum-exp extrema could escape the remaining strategy collisions. This is a numerical continuation method, not a claim that TGV uses randomized or entropy-regularized draft decisions. Temperatures were 0.02, 0.005, 0.001, 0.0002, then zero. The method propagates analytic derivatives through the softened graph; centered finite differences agree within 2.38e-10.

Starting from the best expanded two-product fit, the best **hard-minimax** maximum residual encountered is **0.00136852 logits** at the 0.001-temperature stage. The final zero-temperature refinement returns to 0.00159539, so the saved fit retains the earlier, better hard-minimax result. The first three stages reached their 1,000-evaluation limits. Thus this experiment supplies neither convergence nor a global infeasibility certificate. Independent ordered replay of the retained table agrees exactly.

Reproduction: `python3 scripts/tgv_lowrank_inverse.py --design-prefix earlier --rank 2 --starts 1 --factor-structure both --base-structure additive --refine-from data/tgv/20260916/lowrank-inverse-rank2-earlier-refined-both-additivebase-alljumps.json --temperatures .02,.005,.001,.0002,0`. The saved result has suffix `earlier-refined-both-additivebase-annealed.json` and records each stage's true hard-minimax error and evaluation count. This is a modest improvement in compatible approximation, not original-model recovery.

## Fitting the source-described nonlinear balance term

Re-read the saved current client `current-surface/DraftFeedback-BcwFAnDP.js`. Its descriptions specify squared distance of the team log magic-to-physical ratio from a learned center, plus multiplication by opposing normalized tankiness. The resulting structural expression is:

`L_T = log(M_T/P_T) - center`

`Q_T = balance_coefficient + interaction_coefficient * normalized_tankiness(T)`

`C(B,R) = A_B - A_R + Q_R*L_B^2 - Q_B*L_R^2`.

A combines the linear true-DPM, tankiness, gold-demand, and total-DPM terms. M and P are sums of positive champion contributions. If the tankiness normalization is affine, Q is also additive in champion contributions. The exact normalization remains unpublished; affine normalization is an explicit assumption of this fit.

`scripts/tgv_balance_inverse.py` fits this form directly on the 600-draft domain. It absorbs the unknown center into a rescaling of the positive magic inputs and uses 80 stored parameters. Accordingly, fitted M and P are not claimed to be physical DPM. It also constrains the separately published Blue adjusted-balance effect, Red adjusted-balance effect, and combined linear composition effect from the canonical breakdown—three additional observations beyond the 62 move values.

Sixteen random starts followed by 161 coordinate perturbations produce a maximum residual of **0.00107029 logits across all 65 fitted values**. Analytic derivatives agree with finite differences within 1.08e-10. Independent ordered replay agrees exactly over 1,295,977 states. The separate anchor errors are approximately +0.00001315 for Blue balance, -0.00003174 for Red balance, and +0.00001154 for the linear contribution. The fit is not exact, and no global optimality certificate is claimed. Engine rounding versus double-precision breakdown values also imposes a small numerical consistency limitation, much smaller than the remaining residual.

Reproduction:

- `python3 scripts/tgv_balance_inverse.py --starts 16`
- `python3 scripts/tgv_balance_inverse.py --starts 161 --refine-from data/tgv/20260916/balance-inverse.json`
- `python3 scripts/tgv_replay_bans.py data/tgv/20260916/balance-inverse-refined.json`

This is the most directly source-motivated composition approximation tested here. It still does not identify the original physical channel inputs, training procedure, or arbitrary-draft predictions.

## Gold-input ambiguity persists even when historical game scores are preserved

The previous monotone-transform witness did not retain the physical channel family. A separate linear argument works within the published gold-demand term, subject to explicitly treating its unreleased role/champion/patch inputs as independently variable.

Change only those gold-demand inputs and hold all other score factors fixed. Impose:

1. Zero change to all **53 distinct role/champion cells in the daily pools** on the challenge patch. This preserves every possible daily draft leaf, including flex-role payoffs, and therefore the entire challenge—not only the fitted 62 moves.
2. Zero change in blue-minus-red gold demand for each of the **56 checksum drafts**.
3. Zero change in team-minus-opponent gold demand for every individual scored historical game used by the team-gap summaries. Across the 296 teams with scored games, there are **2,713 game appearances**. Counting duplicate appearances separately is conservative. Preserving every individual score necessarily preserves both published medians.

These are at most **2,822 homogeneous linear equations**. Their precise unknown historical identities cannot increase this rank bound. The gold-input tensor contains 6,055 cells, leaving a nullspace of dimension at least **3,233**. Restricting to the 637 currently declared eligible role/champion pairs across seven patches still gives 4,459 cells and at least **1,637 unconstrained directions**. The restricted count assumes current declared eligibility across the exported patches; the full-tensor count does not.

At most 35 directions correspond to per-role/per-patch constants that cancel in every complete blue-minus-red draft comparison. Thus the restricted bound leaves at least 1,602 directions beyond those constants. Such nonconstant changes affect some unobserved legal scores if the gold coefficient is nonzero on the affected patch. Only the reference patch's nonzero coefficient has been recovered numerically; this score-changing conclusion across every patch is conditional.

This is an exact linear rank bound under the score-input assumptions, not an explicit perturbation vector: the unpublished historical draft identities prevent constructing the full matrix. Sufficiently small perturbations must also remain within any valid input domain. Undisclosed training equations, cross-cell constraints, or additional composition observations could reduce the admissible family. The argument preserves the named numerical observations, not the cryptographic identity of TGV's original factor file.

Reproduce the counts with `python3 scripts/tgv_gold_identifiability.py`. The saved `gold-identifiability-bound.json` records the argument, conservative counts, scope, and assumptions.

## Recovery status

The score decomposition, current logit link, exported strength/matchup/synergy factors, count-based professional comfort curve, and source-described balance structure have been recovered or explicitly characterized. The latest source-motivated local approximation is numerically verified but remains approximate and unidentified beyond its domain. Existing factor-reconstruction tests pass (4/4).

The full original arbitrary-draft model is **not recovered**. The repeated limitation is now mathematical identification: the available numerical observations do not select unique free composition inputs, and the missing training restrictions are not derivable from those observations alone. Continuing to lower a local fit residual would not establish which compatible inputs TGV actually uses. Independent composition measurements or the missing structural/training constraints are needed to resolve that ambiguity; access troubleshooting is not part of this conclusion.

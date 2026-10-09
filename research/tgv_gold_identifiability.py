"""Conservative linear nullspace bound for the unreleased gold-demand inputs.

No historical game identities or coefficient values are invented. Counting every
scored game appearance as a separate constraint overestimates its possible rank.
"""
import json
from pathlib import Path


def main():
    root = Path('data/tgv/20260916')
    runtime = json.loads((root/'current-model-runtime.json').read_text())
    catalog = json.loads((root/'current-model-catalog.json').read_text())
    challenge = json.loads((root/'challenge.json').read_text())['challenge']
    gaps = json.loads((root/'team-draft-gaps.v2.json').read_text())['teams']
    checksums = json.loads((root/'social-betting-strengths.v1.json').read_text())['checksumDrafts']
    roles = dict(top='top', jgl='jungle', mid='middle', bot='bottom', sup='support')
    pool_cells = {(roles[role], cid) for side in challenge['pools'].values()
                  for role, champions in side.items() for cid in champions}
    eligible = {(role, c['cid']) for c in runtime['champions'] for role in c['roles']}
    assert pool_cells <= eligible
    patches = len(catalog['patches'])
    game_appearances = sum(t['gamesScored'] for t in gaps)
    bound = game_appearances + len(checksums) + len(pool_cells)
    all_cells = patches*len(catalog['roles'])*len(runtime['champions'])
    eligible_cells = patches*len(eligible)
    role_constant_gauges = patches*len(catalog['roles'])
    out = dict(
        goldInputCells=all_cells, currentlyEligibleGoldInputCells=eligible_cells,
        frozenDailyPoolCells=len(pool_cells), checksumDrafts=len(checksums),
        scoredHistoricalGameAppearances=game_appearances,
        scoredTeams=sum(t['gamesScored'] > 0 for t in gaps),
        maximumConstraintRank=bound,
        fullTensorNullityLowerBound=all_cells-bound,
        eligibleTensorNullityLowerBound=eligible_cells-bound,
        roleConstantGaugeUpperBound=role_constant_gauges,
        eligibleNonGaugeNullityLowerBound=eligible_cells-bound-role_constant_gauges,
        construction=[
            'Change only the gold-demand channel; hold every other input, coefficient, strength, interaction, and comfort value fixed.',
            'Freeze every role/champion gold cell in either daily pool on the challenge patch. This preserves every possible daily leaf and role-assignment payoff, not merely observed minimizing leaves.',
            'For each checksum draft, impose zero change in blue-minus-red total gold demand.',
            'For every historical game appearance used by a nonempty team-gap summary, impose zero change in team-minus-opponent total gold demand. This preserves each individual game score and consequently both reported medians.',
            'There are at most 2822 homogeneous linear equations. Their identities and dependencies can only reduce rank. A matrix with 4459 eligible columns therefore has nullity at least 1637.',
        ],
        scope='Numerical daily challenge, checksum scores, and current team-gap summaries, holding other exported factor families fixed. The bound does not preserve a cryptographic factor hash.',
        assumptions=[
            'Gold demand is an independently variable role/champion/patch input entering linearly through its team sum, as in the published score description.',
            'The supported gold inputs admit sufficiently small perturbations; undisclosed training equations or cross-cell restrictions could shrink the admissible family.',
            'Currently declared role eligibility is used across the seven exported patches for the restricted count. The full-tensor count does not need this eligibility convention.',
            'A nonconstant gold perturbation changes some unobserved legal draft score when the gold coefficient is nonzero on the affected patch. Only the reference patch coefficient has been numerically recovered, so score-changing conclusions for every patch are conditional.',
        ],
        limitation='This is an exact rank bound under the stated score-input assumptions, not a recovered perturbation vector: the historical draft identities needed to write the full matrix are not available. It proves numerical observations do not uniquely identify those free gold inputs; it does not establish which inputs the undisclosed training process produced.')
    (root/'gold-identifiability-bound.json').write_text(json.dumps(out, indent=2))
    print({k:v for k,v in out.items() if isinstance(v,int)})


if __name__ == '__main__':
    main()

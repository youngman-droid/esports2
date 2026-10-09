"""Reconstruct TGV's publicly exposed draft factors, with explicit missing terms.

No training, network requests, production model writes, or implicit substitution
of the solo-queue champion-board strengths for the full-model strengths.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

ROLES = ('top', 'jungle', 'middle', 'bottom', 'support')


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    e = math.exp(value)
    return e / (1 + e)


def saturating_count(count, amplitude, half_saturation):
    if not math.isfinite(count) or count < 0 or half_saturation <= 0:
        raise ValueError('A nonnegative finite count and positive scale are required')
    return amplitude * count / (count + half_saturation)


def recover_two_point_curve(value_at_one, value_at_two):
    """Solve y=A*n/(n+K) from n=1,2; other counts are independent checks."""
    amplitude = 1 / (2/value_at_two - 1/value_at_one)
    return amplitude, amplitude/value_at_one - 1


class PublicDraftModel:
    def __init__(self, root):
        self.root = Path(root)
        self.catalog = json.loads((self.root/'current-model-catalog.json').read_text())
        if self.catalog['scoreSpace'] != 'logit':
            raise ValueError('Only the verified logit release is supported')
        self.version = self.catalog['modelVersion']
        self.strength = {
            patch: {role: {v['champion']['cid']:v['strength'] for v in rows}
                    for role,rows in insight['roles'].items()}
            for patch,insight in self.catalog['insightsByPatch'].items()}

    @lru_cache(maxsize=865)
    def relations(self, role, champion):
        j = json.loads((self.root/'model-relations/full'/f'{role}-{champion}.json').read_text())
        if j['modelVersion'] != self.version or j['scoreSpace'] != 'logit':
            raise ValueError('Inconsistent model release')
        return {kind: {(r,c):v for c,r,v in values}
                for kind,values in j['relations'][f'{role}:{champion}'].items()}

    def pair(self, kind, role_a, champion_a, role_b, champion_b):
        # Sparse relation omission is not permission to invent a zero.
        return self.relations(role_a,champion_a)[kind][(role_b,champion_b)]

    def components(self, patch, blue, red, blue_comfort=None, red_comfort=None):
        if len(blue)!=5 or len(red)!=5 or len(set(blue+red))!=10:
            raise ValueError('Supply five role-ordered, distinct champions per side')
        if patch not in self.strength:
            raise ValueError('Unsupported patch')
        bc = [0.]*5 if blue_comfort is None else list(blue_comfort)
        rc = [0.]*5 if red_comfort is None else list(red_comfort)
        if len(bc)!=5 or len(rc)!=5 or not all(math.isfinite(x) for x in bc+rc):
            raise ValueError('Invalid comfort values')
        strength = sum(self.strength[patch][r][b]-self.strength[patch][r][d]
                       for r,b,d in zip(ROLES,blue,red))
        match = sum(self.pair('matchups',r,b,s,d)
                    for r,b in zip(ROLES,blue) for s,d in zip(ROLES,red))
        synergy = sum(self.pair('synergies',ROLES[i],blue[i],ROLES[j],blue[j])
                      -self.pair('synergies',ROLES[i],red[i],ROLES[j],red[j])
                      for i in range(5) for j in range(i+1,5))
        return dict(strength=strength,comfort=sum(bc)-sum(rc),matchup=match,synergy=synergy)

    def score(self, patch, blue, red, *, side_bias, composition,
              blue_comfort=None, red_comfort=None):
        """Full score requires supplied side/composition terms; neither defaults to 0."""
        if not math.isfinite(side_bias) or not math.isfinite(composition):
            raise ValueError('Finite side and composition effects are required')
        parts = self.components(patch,blue,red,blue_comfort,red_comfort)
        parts.update(side=side_bias,composition=composition)
        value = sum(parts.values())
        return dict(logit=value,probability=sigmoid(value),components=parts)


def verify_daily(root):
    model = PublicDraftModel(root)
    payload = json.loads((Path(root)/'challenge.json').read_text())
    ch = payload['challenge']; final = ch['puzzle']['finalEvaluation']
    blue = [p['champion']['cid'] for p in final['bluTeam']]
    red = [p['champion']['cid'] for p in final['redTeam']]
    bc = [p['affinity'] for p in final['bluTeam']]
    rc = [p['affinity'] for p in final['redTeam']]
    result = model.score(ch['patch'],blue,red,blue_comfort=bc,red_comfort=rc,
                         side_bias=final['sideBiasScore'],composition=final['compositionScore'])
    targets = dict(strength=final['strengthScore'],comfort=final['modifierScore'],
                   matchup=final['matchupScore'],synergy=final['synergyScore'])
    errors = {k:abs(result['components'][k]-v) for k,v in targets.items()}
    inferred = {x['key']:x['effect']/(x['values']['blue']-x['values']['red'])
                for x in final['compositionFeatures']
                if x['transform']=='linear' and x['values']['blue']!=x['values']['red']}
    return dict(model_version=model.version,patch=ch['patch'],reconstructed=result,
        independent_component_errors=errors,
        review_logit_error=abs(result['logit']-final['reviewBlueDraftAdvantageProbit']),
        engine_logit_error=abs(result['logit']-final['blueDraftAdvantageProbit']),
        review_probability_error=abs(result['probability']-final['reviewBlueWinProbability']),
        inferred_linear_composition_coefficients=inferred,
        supplied_not_recovered=['sideBiasScore for current patch','compositionScore for this draft'],
        limitation='This verifies four recovered factors and the additive/link formula; it does not independently recover composition or other-patch side bias.')

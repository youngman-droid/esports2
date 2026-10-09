"""Match two-game summary constraints to local drafts, retaining both assignments."""
import json
from pathlib import Path
from lol_ticker.tgv_reconstruction import PublicDraftModel


def main():
    root = Path('data/tgv/20260916')
    model = PublicDraftModel(root)
    catalog = model.catalog
    ids = {c['name']:c['cid'] for c in catalog['champions']}
    ranks = {t['teamId']:t for t in json.loads((root/'team-rankings.v3.json').read_text())['teams']}
    gaps = json.loads((root/'gap-central-score-constraints.json').read_text())['teams']
    games = json.loads(Path('data/wpx/draft_comfort_comparison_20260916/inputs.json').read_text())
    out = []
    for t in gaps:
        if t['gamesScored'] != 2:
            continue
        team = ranks[t['teamId']]
        candidates = [g for g in games if g['patch'] in model.strength
            and team['name'] in (g['blue_team'],g['red_team'])
            and g['day'] <= team['lastPlayedAt'][:10]]
        record = dict(teamId=t['teamId'],name=team['name'],lastPlayedAt=team['lastPlayedAt'],
            localSupportedGames=len(candidates),publishedCentralScores=t['centralScores'])
        if len(candidates) == 2:
            known = []
            for g in candidates:
                picks = [[ids[g[side]['picks'][r]] for r in ['top','jng','mid','bot','sup']] for side in ['blue','red']]
                parts = model.components(g['patch'],*picks)
                sign = 1 if g['blue_team']==team['name'] else -1
                subtotal = sign*sum(parts[k] for k in ['strength','matchup','synergy'])
                known.append(dict(gameId=g['id'],day=g['day'],patch=g['patch'],teamSide='blue' if sign==1 else 'red',
                    blue=picks[0],red=picks[1],teamPerspectiveKnownSubtotal=subtotal))
            record['games'] = known
            record['assignments'] = [dict(scoresInGameOrder=values,compositionPlusComfortResiduals=[v-g['teamPerspectiveKnownSubtotal'] for v,g in zip(values,known)])
                for values in [t['centralScores'],list(reversed(t['centralScores']))]]
        out.append(record)
    result = dict(limitation='Candidate matches based on name, covered patch, and TGV last-played date; not verified source game IDs. Each team retains both score assignments. Residual is composition plus comfort, in team perspective; side bias is already removed by the published gap method.',teams=out)
    (root/'two-game-draft-constraints.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

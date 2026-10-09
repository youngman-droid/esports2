"""Conditional composition residuals under historical and export-time comfort counts."""
from collections import defaultdict
import json
from pathlib import Path


def main():
    root=Path('data/tgv/20260916')
    data=json.loads((root/'two-game-draft-constraints.json').read_text())
    rows=json.loads((root/'candidate-historical-comfort-counts.json').read_text())
    export=json.loads((root/'local-player-champion-counts.json').read_text())
    counts={(r['player_id'],r['position'],r['champion']):r['n'] for r in export}
    curve=json.loads((root/'recovered-player-curves.json').read_text())['comfort']
    def comfort(n):return curve['A']*n/(n+curve['K'])
    index=defaultdict(list)
    for row in rows:index[row['game_id']].append(row)
    for team in data['teams']:
        for game in team['games']:
            rs=index[game['gameId']]
            assert len(rs)==10 and sum(r['team']==team['name'] for r in rs)==5
            game['comfortScenarios']={
                'historical':sum((1 if r['team']==team['name'] else -1)*comfort(r['before_game_count']) for r in rs),
                'exportTime':sum((1 if r['team']==team['name'] else -1)*comfort(counts.get((r['player_id'],r['position'],r['champion']),0)) for r in rs)}
        for assignment in team['assignments']:
            assignment['conditionalComposition']={scenario:[residual-game['comfortScenarios'][scenario]
                for residual,game in zip(assignment['compositionPlusComfortResiduals'],team['games'])]
                for scenario in ['historical','exportTime']}
    data['limitation']+=' Comfort timing in TGV draft gaps is not published. Both before-game and export-time count scenarios are retained; neither proves the actual composition term. Local source alignment and curve applicability to historical gaps are assumptions.'
    (root/'candidate-composition-scenarios.json').write_text(json.dumps(data,indent=2))
    for team in data['teams']:
        print(team['name'])
        print('Comfort scenarios',[g['comfortScenarios'] for g in team['games']])
        print('Conditional composition',[a['conditionalComposition'] for a in team['assignments']])


if __name__=='__main__':main()

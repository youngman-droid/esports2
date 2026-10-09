"""Historical input-only diagnostic. Writes only stdout. Never imports feed_backfill.
Run: python3 docs/research-2026-09-11/reproduce_hp_clock.py
Requires project numpy/psycopg dependencies and network for archived opening windows.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import psycopg
from psycopg.rows import dict_row
from lol_ticker.config import PG_DSN, REPO_ROOT
from lol_ticker.live import FEED, _get, _ts

cutoff = '2026-09-03'  # registry-consumed history only; no outcomes selected
z = np.load(Path(REPO_ROOT) / 'data/wpx/states.npz')
names, X, gid, seq, t = z['names'].tolist(), z['X'], z['gid'], z['seq'], z['t']
assert max(z['date']) < cutoff
with psycopg.connect(PG_DSN, row_factory=dict_row,
        options='-c default_transaction_read_only=on -c statement_timeout=15000') as conn:
    records = conn.execute('''
        SELECT fm.esports_game_id,g.game_id,g.date,min(minute) AS first_minute,
               min(ts) AS first_ts
        FROM feed_minutes fm JOIN feed_games fg USING (esports_game_id)
        JOIN golgg_games g ON g.game_id=fg.golgg_game_id
        WHERE g.date < %s
        GROUP BY 1,2,3 HAVING min(minute)>0 ORDER BY g.date
    ''', (cutoff,)).fetchall()
    print(json.dumps({'scope': 'input metadata only', 'read_only':
        conn.execute('SHOW transaction_read_only').fetchone(),
        'cutoff_exclusive':cutoff}, default=str))
    for record in records:
        url = FEED + '/window/' + record['esports_game_id']
        opening = _get(url, allow_empty=True, timeout=15)
        if opening and opening.get('frames'):
            record['endpoint'] = url
            record['opening_t0'] = int(_ts(opening['frames'][0]['rfc460Timestamp']))
            record['hp_shift_earlier_seconds'] = record['first_ts'] - record['opening_t0']
        else:
            record['endpoint_unavailable'] = True
        indices = np.flatnonzero((gid==record['game_id']) & (seq<0) & (t<=120))
        record['cached_rows'] = [dict(t=int(t[i]), **{
            n:float(X[i,names.index(n)]) for n in ('has_hp','hp_pool','lvl_k')}) for i in indices]
        print(json.dumps(record, default=str))
    counts = conn.execute('''
        SELECT count(*) AS games,count(*) FILTER (WHERE m>0) AS late_start_games
        FROM (SELECT fm.esports_game_id,min(minute) m FROM feed_minutes fm
        JOIN feed_games fg USING (esports_game_id)
        JOIN golgg_games g ON g.game_id=fg.golgg_game_id
        WHERE g.date < %s GROUP BY 1) a
    ''', (cutoff,)).fetchone()
    print(json.dumps(counts))

"""Scrape solo-queue lane-vs-lane matchup and teammate-synergy counts (Lolalytics).

Resumable, single-threaded, ~1 request/s. Only champion/role combos seen >=3 times
in pro play on the covered patches. Matchups are fetched once per unordered lane
pair (a's page vs lane j for j >= i); the reverse is 1 - wr. Stops on 403/429 --
no retries past a block. Output: data/sq/lolalytics/<patch>/<lane>-<key>-<what>.json
"""
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import urllib.request

OUT = Path('data/sq/lolalytics')
LANES = ['top', 'jungle', 'middle', 'bottom', 'support']
OE2L = dict(top='top', jng='jungle', mid='middle', bot='bottom', sup='support')
PATCHES = ['16.17', '16.16', '16.15', '16.14', '16.13', '16.12', '16.11', '16.10', '16.9', '16.8']
SPECIAL = {'nunuwillump': 'nunu', 'renataglasc': 'renata', 'monkeyking': 'wukong'}
UA = 'Mozilla/5.0 (research; esports2 private analysis)'


def key(name):
    k = re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower())
    return SPECIAL.get(k, k)


def combos():
    import psycopg
    from lol_ticker.config import PG_DSN
    oe = [p if len(p.split('.')[1]) == 2 else p for p in ('16.08', '16.09', '16.10', '16.11', '16.12', '16.13', '16.14', '16.15', '16.16', '16.17')]
    with psycopg.connect(PG_DSN, options='-c default_transaction_read_only=on') as c:
        rows = c.execute('SELECT p.champion, p.position, count(*) FROM oe_picks p JOIN oe_games g USING (game_id) '
                         'WHERE g.patch = ANY(%s) GROUP BY 1,2 HAVING count(*) >= 3 ORDER BY 3 DESC', (oe,)).fetchall()
    return [(key(ch), OE2L[pos], ch) for ch, pos, _ in rows if pos in OE2L]


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main():
    todo = combos()
    (OUT/'combos.json').parent.mkdir(parents=True, exist_ok=True)
    (OUT/'combos.json').write_text(json.dumps(todo))
    jobs = []
    for patch in PATCHES:
        for k, lane, _ in todo:
            jobs.append((patch, lane, k, 'team', f'ep=build-team&v=1&patch={patch}&c={k}&lane={lane}'))
            for vs in LANES[LANES.index(lane):]:
                jobs.append((patch, lane, k, 'vs-'+vs, f'ep=counter&v=1&patch={patch}&c={k}&lane={lane}&vslane={vs}'))
    done = errors = 0
    for n, (patch, lane, k, what, q) in enumerate(jobs):
        path = OUT/patch/f'{lane}-{k}-{what}.json'
        if path.exists(): continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = fetch(f'https://a1.lolalytics.com/mega/?{q}&tier=emerald_plus&queue=ranked&region=all')
            json.loads(body)      # invalid payloads raise and are not saved
            path.write_bytes(body); done += 1; errors = 0
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                print('blocked', e.code, 'stopping', flush=True); sys.exit(2)
            errors += 1; print('http', e.code, path, flush=True)
        except Exception as e:
            errors += 1; print('err', repr(e)[:80], path, flush=True)
        if errors >= 10:
            print('10 consecutive errors, stopping', flush=True); sys.exit(3)
        if done % 200 == 0: print(f'{n+1}/{len(jobs)} jobs, {done} fetched', flush=True)
        time.sleep(1.0)
    print('complete', len(jobs), flush=True)


if __name__ == '__main__':
    main()

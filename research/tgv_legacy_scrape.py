"""Collect the older model's client-addressed relation JSON, keeping versions separate."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    base = Path('data/tgv/20260916')
    root = base/'legacy-static'
    catalog = json.loads((base/'current-surface/fallback-catalog.json').read_text())
    version = catalog['modelVersion']
    names = [f'model-relations/full/{role}-{champion["cid"]}.json'
             for role in catalog['roles'] for champion in catalog['champions']]
    def get(name):
        path = root/name
        url = 'https://tgv-analytics.com/data/'+name
        try:
            if path.exists():
                raw = path.read_bytes()
            else:
                with urllib.request.urlopen(url,timeout=30) as response:
                    raw = response.read()
                time.sleep(.25)
            obj = json.loads(raw)
            if obj.get('modelVersion') != version:
                raise ValueError('Wrong model version')
            if obj.get('coverage') != 'full-model-interactions':
                raise ValueError('Unexpected relation coverage')
            path.parent.mkdir(exist_ok=True,parents=True)
            path.write_bytes(raw)
            return dict(path=name,url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        except Exception as error:
            return dict(path=name,url=url,error=str(error))
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(get,names):
            records.append(record)
            if len(records)%50 == 0:
                print(f'{len(records)}/{len(names)} checked; {sum("error" in x for x in records)} errors',flush=True)
    result = dict(modelVersion=version,scoreSpace=catalog['scoreSpace'],
        note='Hashes record downloaded contents, not publisher-supplied integrity checks.',
        assets=records,totalBytes=sum(x.get('bytes',0) for x in records),errors=sum('error' in x for x in records))
    (root/'relation-inventory.json').write_text(json.dumps(result,indent=2))
    print(f'Finished {len(records)} relations, {result["errors"]} errors',flush=True)


if __name__ == '__main__':
    main()

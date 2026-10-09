"""Download only JSON assets enumerated by TGV's public release manifest.

Resumable, sequential, checksum-verified, and deliberately excludes media.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/tgv/20260916')
    args = parser.parse_args()
    root = Path(args.root)
    challenge = json.loads((root/'challenge.json').read_text())
    manifest_raw = (root/'release-manifest.json').read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == challenge['challenge']['model']['release_manifest_sha256']
    manifest = json.loads(manifest_raw)
    base = challenge['model_asset_base_url']
    assets = {k:v for k,v in manifest['assets'].items()
              if k.endswith('.json') and not k.startswith('champion-assets/')}
    records = []
    for index,(name,meta) in enumerate(assets.items()):
        if '..' in Path(name).parts or name.startswith('/'):
            raise ValueError('Unsafe manifest path')
        path = root/name
        raw = path.read_bytes() if path.exists() else None
        if raw is None or hashlib.sha256(raw).hexdigest() != meta['sha256']:
            request = urllib.request.Request(base+'/'+name,headers={'User-Agent':'esports2-model-research/1.0'})
            with urllib.request.urlopen(request,timeout=40) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != meta['sha256'] or len(raw) != meta['bytes']:
                raise ValueError('Asset integrity failure: '+name)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
            time.sleep(.25)
        records.append(dict(path=name,bytes=len(raw),sha256=meta['sha256'],url=base+'/'+name))
        if (index+1)%25 == 0: print(f'{index+1}/{len(assets)} JSON assets verified',flush=True)
    (root/'scrape-manifest.json').write_text(json.dumps(dict(model_version=manifest['model_version'],
        assets=records,total_bytes=sum(r['bytes'] for r in records)),indent=2))
    print(f'Complete: {len(records)} assets, {sum(r["bytes"] for r in records)} bytes',flush=True)


if __name__=='__main__': main()

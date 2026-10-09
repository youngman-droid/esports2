"""Join sourced Leaguepedia identities to OE IDs using exact hashed wiki aliases.

No name-only joins: Oracle's IDs are the first 31 MD5 hex characters of a
Leaguepedia page key. Exact canonical page hashes take precedence over recycled display names.
Ambiguous redirect hashes are left unresolved. Birthdate
facts and alias data derive from Leaguepedia (CC BY-SA 3.0), via its public
birth-year tables and GPTilt's lol-esports-entities directory.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote


def oe_id(page):
    return "oe:player:" + hashlib.md5(page.encode()).hexdigest()[:31]


def build(directory=Path('data/player_ratings'), sources=Path('data/oe')):
    data = json.loads((directory / 'directory.json').read_text())
    by_hash = defaultdict(set)
    for person in data['people']:
        by_hash[oe_id(person['person_id'])].add(person['person_id'])
    canonical_hashes = set(by_hash)
    for alias in data['aliases']:
        if alias['entity_type'] == 'person' and alias['alias_type'] != 'real_name' and oe_id(alias['alias']) not in canonical_hashes:
            by_hash[oe_id(alias['alias'])].add(alias['entity_id'])
    births = {}
    for path in sorted(directory.glob('births-*.json')):
        for row in json.loads(path.read_text()):
            page = unquote(row['source_url'].split('/wiki/', 1)[1]).replace('_', ' ')
            identities = by_hash.get(oe_id(page), {page})
            if len(identities) == 1:
                births[next(iter(identities))] = row
    observed = set()
    for path in sources.glob('oe_*.csv'):
        with path.open(newline='', encoding='utf-8-sig') as handle:
            observed.update(row['playerid'] for row in csv.DictReader(handle) if row.get('position') in ('top','jng','mid','bot','sup') and row.get('playerid'))
    birthdays, aliases, provenance = {}, {}, {}
    resolved = defaultdict(list)
    for source_id in observed:
        identities = by_hash.get(source_id, set())
        if len(identities) != 1:
            continue
        identity = next(iter(identities))
        canonical = oe_id(identity)
        resolved[canonical].append(source_id)
        if source_id != canonical:
            aliases[source_id] = canonical
        if identity in births:
            row = births[identity]
            birthdays[canonical] = row['birthday']
            provenance[canonical] = {**row, 'wiki_page': identity, 'identity_match': 'Exact MD5 of canonical wiki page or unambiguous published redirect', 'license': 'CC BY-SA 3.0'}
    for filename, payload in [('birthdays.json', birthdays), ('identity_aliases.json', aliases), ('birthday_provenance.json', provenance)]:
        (directory / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'observed_ids':len(observed),'resolved_ids':sum(map(len,resolved.values())), 'birthdates':len(birthdays),'redirect_aliases':len(aliases)},indent=2))

if __name__ == '__main__':
    build()

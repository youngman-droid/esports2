"""Fetch official Oracle's Elixir archives atomically, preserving existing years.
Source: https://oracleselixir.com/tools/downloads
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import urllib.request

FILES = {
2014: "12syQsRH2QnKrQZTQQ6G5zyVeTG2pAYvu",
2015: "1qyckLuw0-hJM8XqFhlV9l1xAbr3H78T_",
2016: "1muyfpaIqk8_0BFkgLCWXDGNgWSXoPBwG",
2017: "11fx3nNjSYB0X8vKxLAbYOrS2Bu6avm9A",
2018: "1GsNetJQOMx0QJ6_FN8M1kwGvU_GPPcPZ",
2019: "11eKtScnZcpfZcD3w3UrD7nnpfLHvj9_t",
2020: "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
2021: "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
}

def fetch_year(year, directory):
    target = directory / f"oe_{year}.csv"
    if target.exists():
        print(f"{year}: existing archive retained", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={FILES[year]}&export=download&confirm=t"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 esports2 historical research"})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=90) as response, tempfile.NamedTemporaryFile(dir=directory, suffix=".download", delete=False) as output:
            temporary = Path(output.name)
            shutil.copyfileobj(response, output)
        with temporary.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not {"gameid", "date", "playername", "totalgold", "result"} <= set(reader.fieldnames or []):
                raise ValueError(f"{year}: response is not an OE match CSV")
            count = sum(1 for _ in reader)
        if not count:
            raise ValueError(f"{year}: empty archive")
        temporary.replace(target)
        print(f"{year}: {count:,} rows, {target.stat().st_size:,} bytes", flush=True)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/oe"))
    args = parser.parse_args()
    import time
    def retry(year):
        for attempt in range(3):
            try:
                return fetch_year(year, args.directory)
            except (OSError, ValueError) as error:
                print(f"{year}: attempt {attempt + 1} failed: {error}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(retry, FILES))

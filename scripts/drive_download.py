"""Download a (large) public Google Drive file, handling the virus-scan confirm page.
Usage: python3 scripts/drive_download.py <file_id> <out_path>"""
import html, re, sys, urllib.parse, urllib.request, http.cookiejar

def main(file_id, out):
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", "Mozilla/5.0 lol-ticker")]
    url = "https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t" % file_id
    data = op.open(url, timeout=120).read()
    if data[:200].lstrip().lower().startswith(b"<!doctype html") or b"<form" in data[:4000]:
        page = data.decode("utf-8", "replace")
        m = re.search(r'<form[^>]*action="([^"]+)"', page)
        action = html.unescape(m.group(1)) if m else url
        fields = dict((n, html.unescape(v)) for n, v in re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page))
        fields.setdefault("id", file_id); fields.setdefault("export", "download"); fields.setdefault("confirm", "t")
        url2 = action + ("&" if "?" in action else "?") + urllib.parse.urlencode(fields)
        data = op.open(url2, timeout=600).read()
    with open(out, "wb") as f:
        f.write(data)
    print("downloaded %d bytes -> %s" % (len(data), out))
    return 0 if len(data) > 1_000_000 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

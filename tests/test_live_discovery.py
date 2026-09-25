"""Live game discovery: getLive first, then a feed probe for schedule games
whose state Riot ops never flip (LPL)."""
import datetime as dt

from lol_ticker import live


def _iso(ts):
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fake_get(now, schedule_state, game_states, feed_frames):
    calls = []
    lpl = {"type": "match", "state": schedule_state, "startTime": _iso(now - 600),
           "league": {"name": "LPL"},
           "match": {"id": "m1", "strategy": {"count": 5},
                     "teams": [{"id": "A", "name": "WE", "result": {"gameWins": 1}},
                               {"id": "B", "name": "JDG", "result": {"gameWins": 0}}]}}
    old = dict(lpl, startTime=_iso(now - 6 * 3600), match=dict(lpl["match"], id="m0"))

    def get(url, params=None, key=False, allow_empty=False, timeout=30):
        calls.append((url.rsplit("/", 1)[-1], params))
        if url.endswith("getLive"):
            return {"data": {"schedule": {"events": []}}}
        if url.endswith("getSchedule"):
            return {"data": {"schedule": {"events": [old, lpl]}}}
        if url.endswith("getEventDetails"):
            games = [{"id": "g%d" % (i + 1), "number": i + 1, "state": st,
                      "teams": [{"id": "B", "side": "blue"}, {"id": "A", "side": "red"}]}
                     for i, st in enumerate(game_states)]
            return {"data": {"event": dict(lpl, match=dict(lpl["match"], games=games))}}
        if "/window/" in url:
            gid = url.rsplit("/", 1)[-1]
            frames = feed_frames.get(gid)
            return {"frames": frames} if frames else None
        raise AssertionError(url)
    return get, calls


def test_probe_finds_unflagged_lpl_game(monkeypatch):
    now = 1_800_000_000.0
    monkeypatch.setattr(live, "_details_cache", {})
    get, calls = _fake_get(now, "completed", ["completed", "inProgress", "unstarted"],
                           {"g1": [{"gameState": "finished"}], "g2": [{"gameState": "in_game"}]})
    monkeypatch.setattr(live, "_get", get)
    monkeypatch.setattr(live.time, "time", lambda: now)
    games = live.live_games()
    assert [g["game_id"] for g in games] == ["g2"]
    g = games[0]
    assert g["teams"] == ["JDG", "WE"] and g["team_ids"] == ["B", "A"]   # blue first
    assert g["wins"] == [0, 1] and g["number"] == 2 and g["best_of"] == 5
    assert g["discovered_by"] == "feed_probe" and g["sides_known"]
    # the 6-hour-old match is outside the probe window: one details call only
    assert [c for c in calls if c[0] == "getEventDetails"] == [("getEventDetails", {"hl": "en-US", "id": "m1"})]
    # completed game 1 is never probed; the probe asks for a 10 s-aligned window
    probes = [c for c in calls if c[0] in ("g1", "g2", "g3")]
    assert [c[0] for c in probes] == ["g2"]
    assert int(live._ts(probes[0][1]["startingTime"])) % 10 == 0


def test_probe_empty_before_load_in(monkeypatch):
    now = 1_800_000_000.0
    monkeypatch.setattr(live, "_details_cache", {})
    get, calls = _fake_get(now, "unstarted", ["unstarted"], {})
    monkeypatch.setattr(live, "_get", get)
    monkeypatch.setattr(live.time, "time", lambda: now)
    assert live.live_games() == []
    assert live.live_games(probe=False) == []
    assert sum(1 for c in calls if c[0] == "getSchedule") == 1


def test_finished_frames_are_not_live(monkeypatch):
    now = 1_800_000_000.0
    monkeypatch.setattr(live, "_details_cache", {})
    get, _ = _fake_get(now, "unstarted", ["inProgress"], {"g1": [{"gameState": "finished"}]})
    monkeypatch.setattr(live, "_get", get)
    monkeypatch.setattr(live.time, "time", lambda: now)
    assert live.live_games() == []

"""Live win-probability estimate from the LoL Esports live-stats feed.

python3 -m lol_ticker live            # auto-pick the in-progress game
python3 -m lol_ticker live <game_id>  # explicit lolesports game id
"""
import datetime as dt
import json
import re
import logging
import math
import time
import urllib.parse
import urllib.request
import urllib.error

import numpy as np

from . import config

log = logging.getLogger("live")
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"   # public key used by lolesports.com
FEED = "https://feed.lolesports.com/livestats/v1"
CHAMP_ALIASES = {"MonkeyKing": "Wukong", "Ksante": "K'Sante", "KSante": "K'Sante", "Wukong": "Wukong",
                 "RenataGlasc": "Renata Glasc", "Renata": "Renata Glasc", "JarvanIV": "Jarvan IV",
                 "DrMundo": "Dr. Mundo", "TahmKench": "Tahm Kench", "TwistedFate": "Twisted Fate",
                 "MissFortune": "Miss Fortune", "XinZhao": "Xin Zhao", "LeeSin": "Lee Sin",
                 "MasterYi": "Master Yi", "AurelionSol": "Aurelion Sol", "RekSai": "Rek'Sai",
                 "KogMaw": "Kog'Maw", "Chogath": "Cho'Gath", "Khazix": "Kha'Zix", "Velkoz": "Vel'Koz",
                 "Kaisa": "Kai'Sa", "Belveth": "Bel'Veth", "Nunu": "Nunu & Willump", "FiddleSticks": "Fiddlesticks"}


def _get(url, params=None, key=False, allow_empty=False, timeout=30):
    """GET JSON. The live-stats feed answers 200 with an EMPTY body when no
    frames exist for the requested window (too recent, not started, or the
    game flipped) and 404 before the game has started streaming; with
    allow_empty both return None instead of raising."""
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "lol-ticker-live/1.0",
                                               **({"x-api-key": API_KEY} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        if allow_empty and e.code in (400, 404):
            return None   # 404: not streaming yet; 400: requested window too recent
        raise
    if not body.strip():
        if allow_empty:
            return None
        raise RuntimeError("empty response from %s" % url.split("?")[0])
    return json.loads(body)


def live_games():
    d = _get("https://esports-api.lolesports.com/persisted/gw/getLive", {"hl": "en-US"}, key=True)
    out = []
    for e in d.get("data", {}).get("schedule", {}).get("events", []):
        m = e.get("match", {})
        best_of = (m.get("strategy") or {}).get("count")
        mteams = m.get("teams", [])
        for g in m.get("games", []):
            if g.get("state") == "inProgress":
                # The match lists teams in schedule order, NOT by side; each game entry
                # carries [{id, side}] — order blue first so everything downstream
                # (model P(blue), market orientation, labels) refers to the real sides.
                side = {t.get("id"): t.get("side") for t in (g.get("teams") or [])}
                ordered = sorted(mteams, key=lambda t: 0 if side.get(t.get("id")) == "blue" else (1 if side.get(t.get("id")) == "red" else 2))
                wins = [((t.get("result") or {}).get("gameWins") or 0) for t in ordered]
                # deciding map: both teams one win away -> the match-winner market IS the map market
                deciding = bool(best_of and len(wins) == 2 and wins[0] == wins[1] == (best_of - 1) // 2)
                out.append({"game_id": g["id"], "league": e.get("league", {}).get("name"),
                            "teams": [t.get("name") for t in ordered], "team_ids": [t.get("id") for t in ordered],
                            "number": g.get("number"), "best_of": best_of, "wins": wins, "deciding": deciding,
                            "sides_known": all(side.get(t.get("id")) in ("blue", "red") for t in mteams)})
    return out


def _iso(ts):
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _hp_block(bt, rt):
    """HP/level model features from window participants.

    Mirrors the historical ``_hp_feats`` contract: the feed omits health in
    the first minutes of a game, and training marks those states has_hp=0
    with zeroed features — so live must do the same instead of fabricating
    full-health values the model never saw in training.  The display arrays
    keep the full-health assumption for the scoreboard.
    """
    have = all(p.get("maxHealth") for p in bt["participants"] + rt["participants"])
    hpb = [(p["currentHealth"] / p["maxHealth"]) if p.get("maxHealth") else 1.0
           for p in bt["participants"]]
    hpr = [(p["currentHealth"] / p["maxHealth"]) if p.get("maxHealth") else 1.0
           for p in rt["participants"]]
    lvl_k = (sum(p.get("level", 0) for p in bt["participants"])
             - sum(p.get("level", 0) for p in rt["participants"])) / 5.0
    return {
        "hp_pool": (sum(hpb) - sum(hpr)) if have else 0.0,
        "hp_low_b": float(sum(1 for x in hpb if x < 0.3)) if have else 0.0,
        "hp_low_r": float(sum(1 for x in hpr if x < 0.3)) if have else 0.0,
        "lvl_k": lvl_k if have else 0.0,
        "has_hp": 1.0 if have else 0.0,
        "hp_blue": [round(x, 2) for x in hpb], "hp_red": [round(x, 2) for x in hpr],
    }


def fetch_state(game_id, item_gold=None):
    """Current game state from the feed (window + details), plus 2-min-earlier gold."""
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    start_w = _get(FEED + "/window/%s" % game_id)          # opening frames -> game start
    md = start_w["gameMetadata"]
    t0 = _ts(start_w["frames"][0]["rfc460Timestamp"])
    def frames_at(delay):
        st = int((now - delay) // 10 * 10)
        w = _get(FEED + "/window/%s" % game_id, {"startingTime": _iso(st)})
        dd = _get(FEED + "/details/%s" % game_id, {"startingTime": _iso(st)})
        return w, dd
    w, dd = frames_at(60)
    w2, _ = frames_at(180)
    f = w["frames"][-1]; f2 = w2["frames"][-1]
    # paused time: frames with gameState 'paused' are sparse; estimate clock from frame ts
    clock = _ts(f["rfc460Timestamp"]) - t0
    paused = sum(10 for fr in w.get("frames", []) if fr.get("gameState") == "paused")
    bt, rt = f["blueTeam"], f["redTeam"]
    bt2, rt2 = f2["blueTeam"], f2["redTeam"]
    roles = ["top", "jng", "mid", "bot", "sup"]
    gold_role = [(bt["participants"][i]["totalGold"] - rt["participants"][i]["totalGold"]) / 1000.0 for i in range(5)]
    det = dd["frames"][-1]["participants"]
    items_done = 0; item_gold_sum = 0.0
    if item_gold is not None:
        for p in det:
            sgn = 1 if p["participantId"] <= 5 else -1
            for it in p.get("items", []):
                g = item_gold.get(int(it), 0)
                item_gold_sum += sgn * g
                if g >= 2200:
                    items_done += sgn
    champs_b = [p["championId"] for p in md["blueTeamMetadata"]["participantMetadata"]]
    champs_r = [p["championId"] for p in md["redTeamMetadata"]["participantMetadata"]]
    # Health belongs to the window payload.  Details participants contain
    # items and combat stats but no currentHealth field, so reading deaths
    # from ``det`` silently reported everybody alive.
    dead_b = sum(1 for p in bt["participants"] if p.get("currentHealth", 1) <= 0)
    dead_r = sum(1 for p in rt["participants"] if p.get("currentHealth", 1) <= 0)
    bd, rd = bt.get("dragons", []), rt.get("dragons", [])
    elemental_b = [d for d in bd if str(d).lower() != "elder"]
    elemental_r = [d for d in rd if str(d).lower() != "elder"]
    elder_b = sum(str(d).lower() == "elder" for d in bd)
    elder_r = sum(str(d).lower() == "elder" for d in rd)
    state = {
        "game_id": game_id, "patch": md.get("patchVersion"), "clock_s": int(clock), "t_min": clock / 60.0,
        "blue": md["blueTeamMetadata"].get("esportsTeamId"), "red": md["redTeamMetadata"].get("esportsTeamId"),
        "blue_players": [p["summonerName"] for p in md["blueTeamMetadata"]["participantMetadata"]],
        "red_players": [p["summonerName"] for p in md["redTeamMetadata"]["participantMetadata"]],
        "blue_champs": [CHAMP_ALIASES.get(c, c) for c in champs_b], "red_champs": [CHAMP_ALIASES.get(c, c) for c in champs_r],
        "gold_blue": bt["totalGold"], "gold_red": rt["totalGold"],
        "gold_diff_k": (bt["totalGold"] - rt["totalGold"]) / 1000.0,
        "gold_diff_prev_k": (bt2["totalGold"] - rt2["totalGold"]) / 1000.0,
        "gold_role": gold_role,
        "cs_diff_k": (sum(p["creepScore"] for p in bt["participants"]) - sum(p["creepScore"] for p in rt["participants"])) / 100.0,
        "kills": bt["totalKills"] - rt["totalKills"], "kills_blue": bt["totalKills"], "kills_red": rt["totalKills"],
        "towers": bt["towers"] - rt["towers"], "towers_blue": bt["towers"], "towers_red": rt["towers"],
        "inhibs": bt["inhibitors"] - rt["inhibitors"], "inhib_blue": bt["inhibitors"], "inhib_red": rt["inhibitors"],
        "barons": bt["barons"] - rt["barons"], "barons_blue": bt["barons"], "barons_red": rt["barons"],
        "dragons": len(elemental_b) - len(elemental_r),
        "drag_blue": len(elemental_b), "drag_red": len(elemental_r),
        "soul_blue": len(elemental_b) >= 4, "soul_red": len(elemental_r) >= 4,
        "soul": int(len(elemental_b) >= 4) - int(len(elemental_r) >= 4),
        "dragon_types_blue": bd, "dragon_types_red": rd,
        "elders": elder_b - elder_r, "elders_blue": elder_b, "elders_red": elder_r,
        "dead_blue": dead_b, "dead_red": dead_r,
        "items_done_diff": items_done, "item_gold_diff_k": item_gold_sum / 1000.0,
        **_hp_block(bt, rt),
    }
    return state


def estimate(conn, game_id=None, priors=None, teams=None, team_ids=None):
    from . import wpx
    if game_id is None:
        lg = live_games()
        if not lg:
            return {"error": "no in-progress game on the LoL Esports schedule"}
        game_id = lg[0]["game_id"]
        meta = lg[0]
        teams = teams or lg[0].get("teams")
        team_ids = team_ids or lg[0].get("team_ids")
    else:
        meta = {"game_id": game_id}
    item_gold = {r["item_id"]: (r["gold"] or 0) for r in conn.execute("SELECT item_id, gold FROM golgg_items")}
    st = fetch_state(game_id, item_gold)
    # the feed is the authority on sides: reorder the schedule's team names
    # before roster matching when they disagree (the series path does the same)
    if teams and team_ids and len(team_ids) == 2 and st.get("blue") == team_ids[1]:
        teams = list(teams)[::-1]
    roster = pelo_adj = None
    if teams and priors:
        priors = lineup_adjusted_priors(conn, teams, priors,
                                        st.get("blue_players"), st.get("red_players"))
        roster = priors.get("roster")
        pelo_adj = priors.get("pelo_adjustment")
    st.update({k: v for k, v in (priors or {}).items() if k in PRIOR_KEYS})
    pred = wpx.predict_live(st, st["blue_champs"], st["red_champs"])
    # plain production-style estimate (no champion terms) for reference
    pred_nochamp = wpx.predict_live(st, (), ())
    return {"meta": meta, "state": st, "p_blue": pred["p_blue"], "p_blue_no_champ": pred_nochamp["p_blue"],
            "roster": roster, "pelo_adjustment": pelo_adj,
            "unknown_champions": pred["unknown_champions"]}


# ----------------------------------------------------------- market lookup

def _kalshi_markets(series):
    """All open markets of a Kalshi series (cursor-paged)."""
    out, cur = [], None
    for _ in range(20):
        p = {"series_ticker": series, "status": "open", "limit": 200}
        if cur:
            p["cursor"] = cur
        d = _get("https://api.elections.kalshi.com/trade-api/v2/markets", p, timeout=8)
        out.extend(d.get("markets", []))
        cur = d.get("cursor")
        if not cur:
            break
    return out


def _gamma_events(tag):
    """All open Gamma events for a tag. Gamma silently caps limit at 100 -> page by offset."""
    out = []
    for off in range(0, 2000, 100):
        evs = _get("https://gamma-api.polymarket.com/events",
                   {"tag_slug": tag, "closed": "false", "limit": 100, "offset": off}, timeout=8)
        out.extend(evs or [])
        if not evs or len(evs) < 100:
            break
    return out


_resolved = {}   # game key -> {"ts", "complete", "kalshi": {norm: ticker}, "kalshi_src", "pm": {norm: token}, "pm_src", "pm_slug", "pm_fallback": {norm: price}}


def resolve_markets(teams, game_num, deciding=False):
    """Find the exchange markets for this map once (slow scans: Kalshi series
    listing + all open Gamma LoL events). Returns a dict of ids to quote later."""
    from .draft import norm_team
    nb, nr = norm_team(teams[0]), norm_team(teams[1])
    res = {"kalshi": {}, "pm": {}, "pm_fallback": {}, "kalshi_src": None, "pm_src": None, "pm_slug": None}
    try:
        for m in _kalshi_markets("KXLOLMAP"):
            sub = norm_team(m.get("yes_sub_title") or "")
            if (re.search(r"\bmap\s+%d\b" % game_num, m.get("title") or "", re.I)
                    and sub in (nb, nr)):
                res["kalshi"][sub] = m["ticker"]; res["kalshi_src"] = "map"
        if deciding and not res["kalshi"]:
            # match-winner markets: titles vary ("Will X win the A vs. B match?" / "X wins"),
            # so group by event and require both teams among the event's yes_sub_titles
            by_ev = {}
            for m in _kalshi_markets("KXLOLGAME"):
                by_ev.setdefault(m.get("event_ticker"), []).append(m)
            for ev, ms in by_ev.items():
                subs = {norm_team(m.get("yes_sub_title") or ""): m for m in ms}
                if nb in subs and nr in subs:
                    for sub in (nb, nr):
                        res["kalshi"][sub] = subs[sub]["ticker"]
                    res["kalshi_src"] = "series"
                    break
    except Exception as e:
        log.warning("kalshi lookup failed: %s", e)
    try:
        series_cands = []
        for e in _gamma_events("league-of-legends"):
            for m in e.get("markets", []):
                q = m.get("question") or ""
                if m.get("closed") or m.get("acceptingOrders") is False:
                    continue  # stale duplicate from an earlier meeting of the same teams
                outs = json.loads(m.get("outcomes") or "[]")
                no = [norm_team(o) for o in outs]
                if not (nb in no and nr in no):
                    continue
                if "Game %d Winner" % game_num in q:
                    series_cands.insert(0, (m, no, "map"))
                elif deciding and re.search(r"\(BO\d\)", q):
                    series_cands.append((m, no, "series"))
        if series_cands:
            m, no, src = series_cands[0]
            toks = json.loads(m.get("clobTokenIds") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
            for n, t in zip(no, toks):
                res["pm"][n] = t
            for n, pr in zip(no, prices):
                res["pm_fallback"][n] = float(pr)
            res["pm_src"] = src; res["pm_slug"] = m.get("slug")
    except Exception as e:
        log.warning("polymarket lookup failed: %s", e)
    res["complete"] = bool(res["kalshi"]) and bool(res["pm"])
    return res


def quote_markets(res, teams):
    """Fast live quotes for resolved markets: Kalshi GET /markets/{ticker},
    Polymarket CLOB /book midpoints. Blue-oriented by team order."""
    from .draft import norm_team
    nb, nr = norm_team(teams[0]), norm_team(teams[1])
    out = {}
    k = {}
    for sub, tk in res.get("kalshi", {}).items():
        try:
            m = _get("https://api.elections.kalshi.com/trade-api/v2/markets/%s" % tk, timeout=8)["market"]
            bid, ask = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
            status, result = m.get("status"), m.get("result")
            if result in ("yes", "no"):
                # finalized: the book is empty (0.00 / 1.00) -> a midpoint would read 50%; use the settlement
                k[sub] = {"ticker": tk, "mid": 1.0 if result == "yes" else 0.0, "bid": None, "ask": None, "settled": True, "status": status}
            elif bid is not None and ask is not None and not (float(bid) <= 0.0 and float(ask) >= 1.0):
                k[sub] = {"ticker": tk, "mid": (float(bid) + float(ask)) / 2, "bid": float(bid), "ask": float(ask), "status": status}
            elif m.get("last_price_dollars") is not None and status not in ("active", "open"):
                # closed but not yet finalized: empty book, last trade is the best estimate
                k[sub] = {"ticker": tk, "mid": float(m["last_price_dollars"]), "bid": None, "ask": None, "last": True, "status": status}
        except Exception as e:
            log.warning("kalshi quote failed %s: %s", tk, e)
    if k:
        pb, pr = k.get(nb), k.get(nr)
        p = (pb["mid"] + (1 - pr["mid"])) / 2 if (pb and pr) else (pb["mid"] if pb else 1 - pr["mid"])
        out["kalshi"] = {"p_blue": round(p, 4), "detail": k, "source": res.get("kalshi_src") or "map",
                         "settled": any(v.get("settled") for v in k.values()), "status": next(iter(k.values())).get("status"),
                         "captured_ts_ms": int(time.time() * 1000)}
    pm = {}
    for n, tok in res.get("pm", {}).items():
        try:
            book = _get("https://clob.polymarket.com/book", {"token_id": tok}, timeout=8)
            bids = [float(b["price"]) for b in book.get("bids", [])]
            asks = [float(a["price"]) for a in book.get("asks", [])]
            if bids and asks:
                pm[n] = {"mid": (max(bids) + min(asks)) / 2, "bid": max(bids), "ask": min(asks),
                         "price": (max(bids) + min(asks)) / 2, "slug": res.get("pm_slug"),
                         "token_id": tok, "stale": False}
            elif bids or asks:
                # one-sided book: after the game is decided the winner token has only bids (~0.999)
                # and the loser only asks (~0.001) -> take the side that exists
                px = max(bids) if bids else min(asks)
                pm[n] = {"mid": px, "bid": max(bids) if bids else None, "ask": min(asks) if asks else None,
                         "price": px, "slug": res.get("pm_slug"), "token_id": tok,
                         "stale": False, "one_sided": True}
        except Exception as e:
            log.warning("clob book failed %s: %s", n, e)
    for n, pr in res.get("pm_fallback", {}).items():
        if n not in pm and n in (nb, nr) and not pm:
            pm[n] = {"price": pr, "slug": res.get("pm_slug"), "stale": True}   # Gamma cache, lags minutes
    if pm:
        pb, pr = pm.get(nb), pm.get(nr)
        p = pb["price"] if pb else 1 - pr["price"]
        out["polymarket"] = {"p_blue": round(p, 4), "detail": pm, "source": res.get("pm_src") or "map",
                             "stale": any(v.get("stale") for v in pm.values()),
                             "settled": all(v.get("one_sided") for v in pm.values()) and len(pm) == 2,
                             "captured_ts_ms": int(time.time() * 1000)}
    return out


def market_prices(conn, teams, game_num, deciding=False):
    """Current Kalshi / Polymarket prices for this map, blue-oriented by team order.
    Resolution (slow scans) is cached per (teams, map); quotes are fetched live.
    `deciding`: series tied one map from the end -> match-winner market is used
    when no per-map market exists (it resolves on this map). Each platform entry
    carries `source` ('map' | 'series')."""
    key = (tuple(teams), game_num, bool(deciding))
    now = time.time()
    r = _resolved.get(key)
    if r is None or (now - r["ts"] > (600 if r["complete"] else 60)):
        r = resolve_markets(teams, game_num, deciding); r["ts"] = now
        _resolved[key] = r
    return quote_markets(r, teams)


# Prior keys forwarded from team_priors into live model states; every caller
# that filters priors for the model must use this tuple.
PRIOR_KEYS = ("elo_oe", "pelo_oe", "form_diff", "elo_gg")


def team_priors(conn, teams):
    """Current post-game Elo priors for [blue, red]; blue minus red.

    ``oe_ratings`` stores the ratings immediately *before* each game.  Reading
    the newest row verbatim therefore leaves every team one result stale.  We
    advance that row by its known outcome here and keep only a short-lived
    cache so a long-running dashboard sees newly completed games.

    ``elo_gg`` comes the same way from gol.gg games (``golgg_games`` pre-game
    Elo maintained by ``wpa``, K=30) and covers teams even while the Oracle's
    Elixir CSV is stale; OE fields fall back to 0 (= even) when only gol.gg
    knows the teams.
    """
    from .draft import norm_team
    now = time.time()
    cached = getattr(team_priors, "_cache", None)
    if not cached or now - cached["at"] > 60:
        rows = conn.execute("""SELECT g.blue_team, g.red_team, g.winner, g.date_utc,
                                      r.elo_blue, r.elo_red, r.pelo_blue, r.pelo_red
                               FROM oe_games g JOIN oe_ratings r ON r.game_id = g.game_id
                               WHERE g.date_utc > extract(epoch from now()) - 120*86400
                                 AND g.winner IS NOT NULL
                               ORDER BY g.date_utc DESC, g.game_id DESC""").fetchall()
        gg_rows = conn.execute("""SELECT blue_team, red_team, winner_side,
                                         elo_blue_pre, elo_red_pre
                                  FROM golgg_games
                                  WHERE date > now() - interval '120 days'
                                    AND winner_side IS NOT NULL
                                    AND elo_blue_pre IS NOT NULL
                                  ORDER BY date DESC, match_id DESC, game_num DESC""").fetchall()
        cached = {"at": now, "rows": rows, "gg_rows": gg_rows}
        team_priors._cache = cached
    rows = cached["rows"]
    vals = {}
    for name in teams:
        nt = norm_team(name)
        recent = []
        for row in rows:
            blue = norm_team(row["blue_team"]) == nt
            red = norm_team(row["red_team"]) == nt
            if not blue and not red:
                continue
            won = 1.0 if row["winner"] == (row["blue_team"] if blue else row["red_team"]) else 0.0
            if len(recent) < 10:
                recent.append(won)
            if nt not in vals:
                elo = float(row["elo_blue"] if blue else row["elo_red"])
                opp = float(row["elo_red"] if blue else row["elo_blue"])
                pelo = float(row["pelo_blue"] if blue else row["pelo_red"])
                popp = float(row["pelo_red"] if blue else row["pelo_blue"])
                exp_team = 1.0 / (1.0 + 10 ** ((opp - elo) / 400.0))
                exp_player = 1.0 / (1.0 + 10 ** ((popp - pelo) / 400.0))
                vals[nt] = [elo + 30.0 * (won - exp_team),
                            pelo + 24.0 * (won - exp_player), 0.5]
        if nt in vals and recent:
            vals[nt][2] = sum(recent) / len(recent)
    gg_vals = {}
    for name in teams:
        nt = norm_team(name)
        for row in cached.get("gg_rows", []):
            blue = norm_team(row["blue_team"]) == nt
            if not blue and norm_team(row["red_team"]) != nt:
                continue
            elo = float(row["elo_blue_pre"] if blue else row["elo_red_pre"])
            opp = float(row["elo_red_pre"] if blue else row["elo_blue_pre"])
            won = 1.0 if row["winner_side"] == ("blue" if blue else "red") else 0.0
            exp = 1.0 / (1.0 + 10 ** ((opp - elo) / 400.0))
            gg_vals[nt] = elo + 30.0 * (won - exp)
            break
    nb, nr = norm_team(teams[0]), norm_team(teams[1])
    oe_found = nb in vals and nr in vals
    gg_found = nb in gg_vals and nr in gg_vals
    missing = [t for t, n in zip(teams, (nb, nr))
               if n not in vals and n not in gg_vals]
    if missing:
        # Usually a sponsor-name mismatch (fix via draft._TEAM_ALIASES);
        # without it the model silently treats the teams as even.
        log.warning("team_priors: no OE/gol.gg rating match for %s "
                    "(normalized %s)", missing,
                    [norm_team(t) for t in missing])
    if not oe_found and not gg_found:
        return {"found": False}
    out = {"found": True, "oe_found": oe_found, "gg_found": gg_found,
           "elo_oe": 0.0, "pelo_oe": 0.0, "form_diff": 0.0, "elo_gg": None}
    if oe_found:
        b, r_ = vals[nb], vals[nr]
        out.update({"elo_oe": (b[0] - r_[0]) / 400.0, "pelo_oe": (b[1] - r_[1]) / 400.0,
                    "form_diff": b[2] - r_[2], "elo_blue": b[0], "elo_red": r_[0],
                    "pelo_blue": b[1], "pelo_red": r_[1]})
    if gg_found:
        out.update({"elo_gg": (gg_vals[nb] - gg_vals[nr]) / 400.0,
                    "gg_elo_blue": gg_vals[nb], "gg_elo_red": gg_vals[nr]})
    return out


# ------------------------------------------------------------ roster awareness
#
# The team prior describes the lineup that played the team's LAST rated game;
# a substitution today makes that prior stale.  Compare the live feed's
# summoner names against that reference roster and flag the difference.

def _norm_player(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def match_lineup(reference_players, live_names):
    """Match feed summoner names ("C9 Blaber") to gol.gg player names.

    A live name matches a reference player when they are equal after
    normalization, equal once the leading team tag token is stripped, or —
    as a last resort for unusual tag joins — when the reference name (4+
    chars, to avoid short-name collisions) is a suffix of the live name.
    Returns (matched_reference_names, new_live_names, missing_reference_names).
    """
    ref = {_norm_player(p): p for p in reference_players if p}
    matched, new = {}, []
    for name in live_names or []:
        tokens = str(name or "").split()
        cands = {_norm_player(name)}
        if len(tokens) > 1:
            cands.add(_norm_player(" ".join(tokens[1:])))
        cands.discard("")
        hit = next((r for r in ref if r in cands), None)
        if hit is None:
            hit = next((r for r in ref
                        if len(r) >= 4 and any(c.endswith(r) for c in cands)),
                       None)
        if hit is not None and hit not in matched:
            matched[hit] = name
        elif name:
            new.append(name)
    missing = [ref[r] for r in ref if r not in matched]
    return sorted(matched.values()), new, missing


def _reference_roster(conn, team):
    """(game_id, date, [player names]) of the team's newest gol.gg game."""
    from .draft import norm_team
    nt = norm_team(team)
    now = time.time()
    cached = getattr(_reference_roster, "_cache", None)
    if not cached or now - cached["at"] > 600:
        cached = {"at": now, "teams": {}, "games": conn.execute(
            """SELECT game_id, blue_team, red_team, date FROM golgg_games
               WHERE date > now() - interval '120 days'
               ORDER BY date DESC, match_id DESC, game_num DESC""").fetchall()}
        _reference_roster._cache = cached
    if nt in cached["teams"]:
        return cached["teams"][nt]
    ref = None
    for g in cached["games"]:
        blue = norm_team(g["blue_team"]) == nt
        if not blue and norm_team(g["red_team"]) != nt:
            continue
        side = "blue" if blue else "red"
        players = [r["player"] for r in conn.execute(
            """SELECT player FROM golgg_players
               WHERE game_id=%s AND side=%s ORDER BY slot""",
            (g["game_id"], side))]
        if players:
            ref = {"game_id": g["game_id"], "date": str(g["date"]),
                   "players": players}
            break
        # newest game has no player rows (scrape gap): try the next one
    cached["teams"][nt] = ref
    return ref


def _player_elos(conn):
    """norm IGN -> (elo, ngames, last_ts) from oe_player_elo; newest wins."""
    now = time.time()
    cached = getattr(_player_elos, "_cache", None)
    if not cached or now - cached["at"] > 600:
        table = {}
        try:
            for r in conn.execute(
                    "SELECT norm_name, elo, ngames, last_ts FROM oe_player_elo"):
                cur = table.get(r["norm_name"])
                if cur is None or (r["last_ts"] or 0) > cur[2]:
                    table[r["norm_name"]] = (float(r["elo"]), int(r["ngames"] or 0),
                                             int(r["last_ts"] or 0))
        except Exception as exc:
            conn.rollback()
            log.warning("player elo table unavailable: %s", exc)
        cached = {"at": now, "table": table}
        _player_elos._cache = cached
    return cached["table"]


def _resolve_lineup(table, names):
    """([elo per resolved player], [unresolved names]) for feed summoner names."""
    vals, unresolved = [], []
    for name in names or []:
        tokens = str(name or "").split()
        cands = [c for c in (_norm_player(name),
                             _norm_player(" ".join(tokens[1:])) if len(tokens) > 1 else "")
                 if c]
        hit = next((table[c] for c in cands if c in table), None)
        if hit is None:
            # tagless joins ("T1Faker"): longest sufficiently-long suffix wins
            suffix = sorted((k for k in table
                             if len(k) >= 4 and any(c.endswith(k) for c in cands)),
                            key=len, reverse=True)
            hit = table[suffix[0]] if suffix else None
        if hit is not None:
            vals.append(hit[0])
        elif name:
            unresolved.append(name)
    return vals, unresolved


def lineup_adjusted_priors(conn, teams, priors, blue_players, red_players):
    """Recompute the player-Elo prior from the lineup actually on the rift.

    Historically ``pelo`` is the mean rating of the five players who played
    each game, so this only corrects the live approximation (which reads the
    team's last lineup).  A side is adjusted when its roster differs from
    the reference roster and at least four of its five names resolve in
    ``oe_player_elo``; unresolved players inherit the side's team pelo (or
    1500).  Other prior channels are left untouched.
    """
    priors = dict(priors or {})
    roster = roster_check(conn, teams, blue_players, red_players)
    priors["roster"] = roster
    table = _player_elos(conn)
    means, info = {}, {}
    for side, names in (("blue", blue_players), ("red", red_players)):
        team_mean = priors.get("pelo_%s" % side)
        rc = roster.get(side) or {}
        needs = (rc.get("changed") or team_mean is None) and bool(names)
        applied = False
        if needs and table:
            vals, unresolved = _resolve_lineup(table, names)
            if len(vals) >= 4:
                fill = team_mean if team_mean is not None else 1500.0
                mean = (sum(vals) + fill * len(unresolved)) / (len(vals) + len(unresolved))
                means[side] = mean
                applied = True
                info[side] = {"applied": True, "lineup_pelo": round(mean, 1),
                              "reference_pelo": (round(team_mean, 1)
                                                 if team_mean is not None else None),
                              "resolved": len(vals), "unresolved": unresolved}
        if not applied:
            means[side] = team_mean
            info[side] = {"applied": False}
    if means["blue"] is not None and means["red"] is not None:
        old = priors.get("pelo_oe")
        priors["pelo_oe"] = (means["blue"] - means["red"]) / 400.0
        if any(v["applied"] for v in info.values()):
            priors["found"] = True
            log.info("lineup-adjusted pelo: %.3f -> %.3f (%s)",
                     old if old is not None else float("nan"),
                     priors["pelo_oe"],
                     {s: v for s, v in info.items() if v["applied"]})
    priors["pelo_adjustment"] = info
    return priors


def roster_check(conn, teams, blue_players, red_players):
    """Per-side lineup comparison against each team's last rated roster."""
    out = {}
    for side, team, names in (("blue", teams[0], blue_players),
                              ("red", teams[1], red_players)):
        ref = _reference_roster(conn, team)
        if not ref or not names:
            out[side] = {"available": False, "team": team}
            continue
        matched, new, missing = match_lineup(ref["players"], names)
        changed = bool(new) and bool(missing)
        out[side] = {"available": True, "team": team, "changed": changed,
                     "matched": len(matched), "new": new, "missing": missing,
                     "reference_game_id": ref["game_id"],
                     "reference_date": ref["date"]}
        if changed:
            log.warning("roster change for %s: %s in, %s out "
                        "(prior reflects gol.gg game %s on %s)",
                        team, new, missing, ref["game_id"], ref["date"])
    return out


# -------------------------------------------------------- 1 Hz frame series

_cache = {}   # game_id -> {"t0", "md", "champs_b", "champs_r", "item_gold", "seen": set(), "markets": (ts, val), "priors"}


def _frame_state(md, f, f_prev, det, item_gold, t0, game_clock_s=None):
    """State dict for one window frame (+ matching details frame)."""
    clock = (_ts(f["rfc460Timestamp"]) - t0) if game_clock_s is None else game_clock_s
    bt, rt = f["blueTeam"], f["redTeam"]
    bt2, rt2 = (f_prev["blueTeam"], f_prev["redTeam"]) if f_prev else (bt, rt)
    gold_role = [(bt["participants"][i]["totalGold"] - rt["participants"][i]["totalGold"]) / 1000.0 for i in range(5)]
    dead_b = sum(1 for p in bt["participants"] if p.get("currentHealth", 1) <= 0)
    dead_r = sum(1 for p in rt["participants"] if p.get("currentHealth", 1) <= 0)
    items_done = 0; item_gold_sum = 0.0
    if det:
        for p in det.get("participants", []):
            sgn = 1 if p["participantId"] <= 5 else -1
            for it in p.get("items", []):
                g = item_gold.get(int(it), 0)
                item_gold_sum += sgn * g
                if g >= 2200:
                    items_done += sgn
    bd, rd = bt.get("dragons", []), rt.get("dragons", [])
    elemental_b = [d for d in bd if str(d).lower() != "elder"]
    elemental_r = [d for d in rd if str(d).lower() != "elder"]
    elder_b = sum(str(d).lower() == "elder" for d in bd)
    elder_r = sum(str(d).lower() == "elder" for d in rd)
    return {
        "ts": int(_ts(f["rfc460Timestamp"])), "clock_s": int(clock), "t_min": clock / 60.0,
        "gold_blue": bt["totalGold"], "gold_red": rt["totalGold"],
        "gold_diff_k": (bt["totalGold"] - rt["totalGold"]) / 1000.0,
        "gold_diff_prev_k": (bt2["totalGold"] - rt2["totalGold"]) / 1000.0,
        "gold_role": gold_role,
        "cs_diff_k": (sum(p["creepScore"] for p in bt["participants"]) - sum(p["creepScore"] for p in rt["participants"])) / 100.0,
        "kills": bt["totalKills"] - rt["totalKills"], "kills_blue": bt["totalKills"], "kills_red": rt["totalKills"],
        "towers": bt["towers"] - rt["towers"], "towers_blue": bt["towers"], "towers_red": rt["towers"],
        "inhibs": bt["inhibitors"] - rt["inhibitors"], "inhib_blue": bt["inhibitors"], "inhib_red": rt["inhibitors"],
        "barons": bt["barons"] - rt["barons"], "barons_blue": bt["barons"], "barons_red": rt["barons"],
        "dragons": len(elemental_b) - len(elemental_r),
        "drag_blue": len(elemental_b), "drag_red": len(elemental_r),
        "soul_blue": len(elemental_b) >= 4, "soul_red": len(elemental_r) >= 4,
        "soul": int(len(elemental_b) >= 4) - int(len(elemental_r) >= 4),
        "dragon_types_blue": bd, "dragon_types_red": rd,
        "elders": elder_b - elder_r, "elders_blue": elder_b, "elders_red": elder_r,
        "dead_blue": dead_b, "dead_red": dead_r,
        "items_done_diff": items_done, "item_gold_diff_k": item_gold_sum / 1000.0,
        **_hp_block(bt, rt),
    }


def _md_champs(md):
    return ([CHAMP_ALIASES.get(p["championId"], p["championId"]) for p in md["blueTeamMetadata"]["participantMetadata"]],
            [CHAMP_ALIASES.get(p["championId"], p["championId"]) for p in md["redTeamMetadata"]["participantMetadata"]])


def _new_cache(conn, md, t0):
    cb, cr = _md_champs(md)
    return {"t0": t0, "md": md,
            "blue_team_id": (md.get("blueTeamMetadata") or {}).get("esportsTeamId"),
            "red_team_id": (md.get("redTeamMetadata") or {}).get("esportsTeamId"),
            "champs_b": cb, "champs_r": cr,
            "item_gold": {r["item_id"]: (r["gold"] or 0) for r in conn.execute("SELECT item_id, gold FROM golgg_items")},
            "item_names": {r["item_id"]: r["name"] for r in conn.execute("SELECT item_id, name FROM golgg_items")},
            "seen": set(), "markets": (0, {}),
            "prev": {"pause_s": 0.0, "last_frame_ts": None, "last_game_state": None,
                     "baron_seeded": False, "baron_counts": None,
                     "last_baron_clock": [None, None],
                     "elder_seeded": False, "elder_counts": None,
                     "last_elder_clock": [None, None],
                     "kill_seeded": False, "last_kill_clock": None}}


def _baron_counts(frame):
    return [int(frame["blueTeam"].get("barons", 0) or 0),
            int(frame["redTeam"].get("barons", 0) or 0)]


def _elder_counts(frame):
    def count(team):
        return sum(str(d).lower() == "elder" for d in team.get("dragons", []))
    return [count(frame["blueTeam"]), count(frame["redTeam"])]


def _total_kills(frame):
    return (int(frame["blueTeam"].get("totalKills", 0) or 0)
            + int(frame["redTeam"].get("totalKills", 0) or 0))


def _baron_probe(game_id, ts):
    """Nearest feed frame to ``ts`` for mid-game Baron-state recovery."""
    w = _get(FEED + "/window/%s" % game_id,
             {"startingTime": _iso(int(ts // 10 * 10))}, allow_empty=True)
    frames = (w or {}).get("frames") or []
    return min(frames, key=lambda f: abs(_ts(f["rfc460Timestamp"]) - ts)) if frames else None


def _seed_recent_objective_clocks(game_id, frame, game_clock, t0,
                                  duration_s, count_fn):
    """Recover a timed-objective acquisition after a mid-game restart.

    Compare the current cumulative counters with a frame just beyond the buff
    duration, then binary-search the counter transition. This is paid once per
    game/process/objective, not on every poll.
    """
    current_ts = _ts(frame["rfc460Timestamp"])
    current = count_fn(frame)
    result = [None, None]
    if not any(current):
        return result
    lower_ts = max(float(t0), current_ts - float(duration_s) - 1.0)
    probes = {}

    def probe(ts):
        key = int(ts // 10 * 10)
        if key not in probes:
            probes[key] = _baron_probe(game_id, ts)
        return probes[key]

    lower = probe(lower_ts)
    lower_counts = count_fn(lower) if lower else [0, 0]
    for side in range(2):
        target = current[side]
        if target <= lower_counts[side]:
            continue
        lo, hi = lower_ts, current_ts
        for _ in range(8):
            if hi - lo <= 10.0:
                break
            mid = (lo + hi) / 2.0
            f_mid = probe(mid)
            if f_mid is None:
                break
            mid_ts = _ts(f_mid["rfc460Timestamp"])
            if count_fn(f_mid)[side] >= target:
                hi = min(hi, mid_ts)
            else:
                lo = max(lo, mid_ts)
        result[side] = max(0.0, float(game_clock) - (current_ts - hi))
    return result


def _seed_recent_baron_clocks(game_id, frame, game_clock, t0):
    return _seed_recent_objective_clocks(
        game_id, frame, game_clock, t0, 180.0, _baron_counts)


def _seed_recent_elder_clocks(game_id, frame, game_clock, t0):
    return _seed_recent_objective_clocks(
        game_id, frame, game_clock, t0, 150.0, _elder_counts)


def _seed_recent_kill_clock(game_id, frame, game_clock, t0):
    """Recover the latest kill time when live scoring joins mid-game.

    ``t_since_kill`` is capped at ten minutes.  Probe that window and locate the
    transition to the current cumulative kill count, so a process restart does
    not incorrectly report ten quiet minutes until the next kill.
    """
    current_ts = _ts(frame["rfc460Timestamp"])
    target = _total_kills(frame)
    if target <= 0:
        return None
    lower_ts = max(float(t0), current_ts - 600.0)
    lower = _baron_probe(game_id, lower_ts)
    if lower is None or _total_kills(lower) >= target:
        return None
    lo, hi = lower_ts, current_ts
    for _ in range(8):
        if hi - lo <= 10.0:
            break
        mid = (lo + hi) / 2.0
        probe = _baron_probe(game_id, mid)
        if probe is None:
            break
        probe_ts = _ts(probe["rfc460Timestamp"])
        if _total_kills(probe) >= target:
            hi = min(hi, probe_ts)
        else:
            lo = max(lo, probe_ts)
    return max(0.0, float(game_clock) - (current_ts - hi))


def _update_timed_objective(prev, frame, game_clock, seeded_clocks,
                            counts_key, clocks_key, seeded_key,
                            duration_s, count_fn):
    counts = count_fn(frame)
    if seeded_clocks is not None:
        prev[clocks_key] = list(seeded_clocks)
        prev[counts_key] = list(counts)
        prev[seeded_key] = True
    old = prev.get(counts_key)
    clocks = prev.setdefault(clocks_key, [None, None])
    if old is not None:
        for side in range(2):
            if counts[side] > old[side]:
                clocks[side] = float(game_clock)
            elif counts[side] < old[side]:
                clocks[side] = None
    prev[counts_key] = counts
    timers = [max(0.0, float(duration_s) - (float(game_clock) - clock))
              if clock is not None and float(game_clock) >= clock else 0.0
              for clock in clocks]
    return timers


def _update_baron_state(prev, frame, game_clock, seeded_clocks=None):
    """Update counters and return blue/red Baron-buff seconds remaining."""
    return _update_timed_objective(
        prev, frame, game_clock, seeded_clocks,
        "baron_counts", "last_baron_clock", "baron_seeded", 180.0, _baron_counts)


def _update_elder_state(prev, frame, game_clock, seeded_clocks=None):
    """Update counters and return blue/red Elder-buff seconds remaining."""
    return _update_timed_objective(
        prev, frame, game_clock, seeded_clocks,
        "elder_counts", "last_elder_clock", "elder_seeded", 150.0, _elder_counts)


def _restart_t0(game_id, ts_hint, t0_old):
    """A remade game reuses its game id: frames restart (team gold back at 2500)
    after a gap. Walk 10 s windows back from ts_hint to the earliest frame of the
    restarted attempt (stop at the gap or at old-attempt frames with real gold)."""
    first_ts = ts_hint
    st = int(ts_hint // 10 * 10) - 10
    for _ in range(90):
        if st <= t0_old:
            break
        w = _get(FEED + "/window/%s" % game_id, {"startingTime": _iso(st)}, allow_empty=True)
        if not w or not w.get("frames"):
            break
        f0 = w["frames"][0]
        if f0["blueTeam"]["totalGold"] + f0["redTeam"]["totalGold"] > 7000:
            break   # frames from the aborted attempt
        first_ts = _ts(f0["rfc460Timestamp"])
        st -= 10
    return first_ts


def estimate_series(conn, game_id, priors=None, since_ts=0, teams=None):
    """Score every new 1 Hz frame since `since_ts`; returns frames + summary.

    Uses the feed's 10-frame windows: one window ~70 s back (latest available),
    one ~190 s back (gold momentum), one details window for items/health.
    When ``teams`` is given, the player-Elo prior is recomputed for the
    lineup actually on the rift (see lineup_adjusted_priors) before scoring.
    """
    from . import wpx
    c = _cache.get(game_id)
    if c is None:
        w0 = _get(FEED + "/window/%s" % game_id, allow_empty=True)
        if not w0 or not w0.get("frames"):
            return {"frames": [], "note": "feed has not started for this game yet", "blue_champs": [], "red_champs": [],
                    "patch": None, "game_start_ts": None}
        c = _new_cache(conn, w0["gameMetadata"], _ts(w0["frames"][0]["rfc460Timestamp"]))
        _cache[game_id] = c
    roster = pelo_adj = None
    if teams and priors:
        md = c["md"]
        priors = lineup_adjusted_priors(
            conn, teams, priors,
            [p.get("summonerName") for p in md["blueTeamMetadata"]["participantMetadata"]],
            [p.get("summonerName") for p in md["redTeamMetadata"]["participantMetadata"]])
        roster = priors.get("roster")
        pelo_adj = priors.get("pelo_adjustment")
    priors = {k: v for k, v in (priors or {}).items() if k in PRIOR_KEYS}
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    w = dd = None
    # The feed serves 10 s windows whose start must be >= ~40 s in the past
    # (probed 2026-08-22: start at now-33s -> HTTP 400, now-43s -> ok), so the
    # freshest frame is ~35-45 s old. Ask for that window first, step back if refused.
    # The feed is patchy at times (whole 10 s windows missing, newest frame 1-2 min
    # behind): walk back from the freshest allowed window until one has frames,
    # but not past the last window we already served (c["last_st"]).
    st_now = int((now - 40) // 10 * 10)
    # don't walk back past what this client already has (since_ts), nor more than 3 min
    floor_st = max(st_now - 180, (int(since_ts) // 10 * 10) if since_ts else 0)
    st = st_now
    while st > floor_st:
        w = _get(FEED + "/window/%s" % game_id, {"startingTime": _iso(st)}, allow_empty=True)
        if w and w.get("frames"):
            dd = _get(FEED + "/details/%s" % game_id, {"startingTime": _iso(st)}, allow_empty=True)
            st_now = st
            break
        st -= 10
    if not w or not w.get("frames"):
        if c.get("last_st"):
            note = "feed is lagging: no new frames in the last %d s" % int(now - c["last_st"] - 10)
        else:
            note = "feed has no frames yet for this game (it starts publishing ~1 min after game start)"
        return {"frames": [], "note": note, "blue_champs": c["champs_b"], "red_champs": c["champs_r"],
                "patch": c["md"].get("patchVersion"), "game_start_ts": int(c["t0"]), "feed_lag_s": int(now - (c.get("last_st") or c["t0"]))}
    c["last_st"] = st_now
    # ---- remake detection: same game id, new draft and/or clock reset ----
    md_now = w.get("gameMetadata") or {}
    f_last = w["frames"][-1]
    gold_sum = f_last["blueTeam"]["totalGold"] + f_last["redTeam"]["totalGold"]
    champs_changed = False
    if md_now.get("blueTeamMetadata"):
        cb_now, cr_now = _md_champs(md_now)
        champs_changed = (cb_now != c["champs_b"] or cr_now != c["champs_r"])
    clock_reset = (_ts(f_last["rfc460Timestamp"]) - c["t0"] > 420 and gold_sum < 7000
                   and f_last.get("gameState") == "in_game")
    if champs_changed or clock_reset:
        t0_new = _restart_t0(game_id, _ts(w["frames"][0]["rfc460Timestamp"]), c["t0"])
        log.info("remake detected for %s (%s): clock re-anchored %s -> %s",
                 game_id, "new draft" if champs_changed else "gold reset", _iso(c["t0"]), _iso(t0_new))
        c = _new_cache(conn, md_now if md_now.get("blueTeamMetadata") else c["md"], t0_new)
        c["last_st"] = st_now
        _cache[game_id] = c
    w_prev = _get(FEED + "/window/%s" % game_id, {"startingTime": _iso(st_now - 120)}, allow_empty=True) or {}
    det_by_ts = {int(_ts(f["rfc460Timestamp"])): f for f in (dd or {}).get("frames", [])}
    prev_by_ts = {int(_ts(f["rfc460Timestamp"])): f for f in w_prev.get("frames", [])
                  if _ts(f["rfc460Timestamp"]) >= c["t0"]}
    prev = c["prev"]
    if not prev.get("kill_seeded") and w.get("frames"):
        seed_frame = w["frames"][0]
        seed_clock = max(0.0, _ts(seed_frame["rfc460Timestamp"]) - c["t0"]
                         - prev.get("pause_s", 0.0))
        prev["last_kill_clock"] = _seed_recent_kill_clock(
            game_id, seed_frame, seed_clock, c["t0"])
        prev["kills"] = _total_kills(seed_frame)
        prev["kill_seeded"] = True
    out = []
    seen_this_call = set()   # the feed emits several sub-second frames; keep one per second
    for f in w.get("frames", []):
        ts = int(_ts(f["rfc460Timestamp"]))
        if ts < c["t0"] or ts <= since_ts or ts in seen_this_call or f.get("gameState") not in ("in_game", "paused", "finished"):
            continue
        seen_this_call.add(ts)
        f_prev = prev_by_ts.get(ts - 120) or (next(iter(prev_by_ts.values()), None) if prev_by_ts else None)
        last_ts = prev.get("last_frame_ts")
        if last_ts is not None and prev.get("last_game_state") == "paused":
            prev["pause_s"] = prev.get("pause_s", 0.0) + max(0, ts - last_ts)
        game_clock = max(0.0, ts - c["t0"] - prev.get("pause_s", 0.0))
        baron_seeded = None
        if not prev.get("baron_seeded"):
            baron_seeded = _seed_recent_baron_clocks(game_id, f, game_clock, c["t0"])
        elder_seeded = None
        if not prev.get("elder_seeded"):
            elder_seeded = _seed_recent_elder_clocks(game_id, f, game_clock, c["t0"])
        baron_timers = _update_baron_state(prev, f, game_clock, baron_seeded)
        elder_timers = _update_elder_state(prev, f, game_clock, elder_seeded)
        st = _frame_state(c["md"], f, f_prev, det_by_ts.get(ts), c["item_gold"], c["t0"], game_clock)
        st["baron_timer_blue_s"] = round(baron_timers[0], 1)
        st["baron_timer_red_s"] = round(baron_timers[1], 1)
        st["baron_active_blue"] = baron_timers[0] > 0.0
        st["baron_active_red"] = baron_timers[1] > 0.0
        st["baron_active"] = int(st["baron_active_blue"]) - int(st["baron_active_red"])
        st["elder_timer_blue_s"] = round(elder_timers[0], 1)
        st["elder_timer_red_s"] = round(elder_timers[1], 1)
        st["elder_active_blue"] = elder_timers[0] > 0.0
        st["elder_active_red"] = elder_timers[1] > 0.0
        st["elder_active"] = int(st["elder_active_blue"]) - int(st["elder_active_red"])
        prev["last_frame_ts"] = ts
        prev["last_game_state"] = f.get("gameState")
        # time since last kill: track total kills across frames
        tk = st["kills_blue"] + st["kills_red"]
        if prev.get("kills") is not None and tk > prev["kills"]:
            prev["last_kill_clock"] = st["clock_s"]
        prev["kills"] = tk
        st["t_since_kill_min"] = min(10.0, (st["clock_s"] - prev["last_kill_clock"]) / 60.0) if prev.get("last_kill_clock") is not None else 10.0
        st.update(priors or {})
        pr = wpx.predict_live(st, c["champs_b"], c["champs_r"])
        p = pr["p_blue"]
        p0 = wpx.predict_live(st, (), ())["p_blue"]
        st["p_blue"] = p; st["p_blue_no_champ"] = p0; st["game_state"] = f.get("gameState")
        st["lo_prior"] = pr["lo_prior"]; st["lo_state"] = pr["lo_state"]; st["lo_champ"] = pr["lo_champ"]; st["lo_time"] = pr["lo_time"]
        st["lo_deaths"] = pr.get("lo_deaths"); st["terminal_state"] = pr.get("terminal_state", False)
        # recorded so the shadow ledger can split scores by champion-effect
        # magnitude (the champ-heavy calibration check is unconfirmed on the
        # backtest; the ledger will answer it prospectively)
        st["lo_champ_state"] = pr.get("lo_champ_state")
        st["lo_baron_active"] = pr.get("lo_baron_active")
        st["lo_elder_active"] = pr.get("lo_elder_active")
        st["model_input_warnings"] = pr.get("input_warnings") or []
        st["model_clipped_inputs"] = pr.get("clipped_inputs") or []
        st["model_reliability"] = pr.get("reliability", "normal")
        st["model_kind"] = pr.get("model_kind")
        out.append(st)
    out.sort(key=lambda x: x["ts"])
    # scoreboard: latest window frame + matching details frame
    last = w["frames"][-1]
    det = det_by_ts.get(int(_ts(last["rfc460Timestamp"]))) or (dd["frames"][-1] if dd and dd.get("frames") else None)
    det_p = {p["participantId"]: p for p in (det or {}).get("participants", [])}
    rows = []
    for side, key, meta_key in (("blue", "blueTeam", "blueTeamMetadata"), ("red", "redTeam", "redTeamMetadata")):
        meta_p = {p["participantId"]: p for p in c["md"][meta_key]["participantMetadata"]}
        for p in last[key]["participants"]:
            pid = p["participantId"]; m = meta_p.get(pid, {}); d = det_p.get(pid, {})
            hp, mhp = p.get("currentHealth"), p.get("maxHealth")
            rows.append({
                "side": side, "pid": pid, "player": m.get("summonerName"), "role": m.get("role"),
                "champion": CHAMP_ALIASES.get(m.get("championId"), m.get("championId")),
                "level": p.get("level"), "k": p.get("kills"), "d": p.get("deaths"), "a": p.get("assists"),
                "cs": p.get("creepScore"), "gold": p.get("totalGold"),
                "hp": hp, "max_hp": mhp, "alive": (hp is None or hp > 0),
                "items": [c["item_names"].get(int(i), str(i)) for i in d.get("items", []) if i],
                "wards": d.get("wardsPlaced"), "kp": d.get("killParticipation"), "dmg_share": d.get("championDamageShare"),
            })
    bt, rt = last["blueTeam"], last["redTeam"]
    totals = {"blue": {k: bt.get(k) for k in ("totalGold", "totalKills", "towers", "inhibitors", "barons")} | {"dragons": bt.get("dragons", [])},
              "red": {k: rt.get(k) for k in ("totalGold", "totalKills", "towers", "inhibitors", "barons")} | {"dragons": rt.get("dragons", [])}}
    return {"frames": out, "blue_champs": c["champs_b"], "red_champs": c["champs_r"],
            "patch": c["md"].get("patchVersion"), "game_start_ts": int(c["t0"]),
            "blue_team_id": c.get("blue_team_id"), "red_team_id": c.get("red_team_id"),
            "scoreboard": rows, "totals": totals, "frame_ts": int(_ts(last["rfc460Timestamp"])),
            "roster": roster, "pelo_adjustment": pelo_adj, "priors_effective": priors,
            "feed_lag_s": int(now - _ts(last["rfc460Timestamp"]))}

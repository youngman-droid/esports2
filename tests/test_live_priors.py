import unittest
import copy
import datetime as dt
from unittest import mock

from lol_ticker import draft, live


class _Conn:
    """Serves the OE-ratings query first, then the gol.gg query."""

    def __init__(self, oe_rows, gg_rows=()):
        self.row_sets = [list(oe_rows), list(gg_rows)]
        self.calls = 0

    def execute(self, _query):
        self.calls += 1
        self._pending = self.row_sets[min(self.calls - 1, 1) % 2]
        return self

    def fetchall(self):
        return self._pending


_OE_ROW = {
    "blue_team": "Alpha", "red_team": "Beta", "winner": "Alpha",
    "date_utc": 1, "elo_blue": 1500.0, "elo_red": 1500.0,
    "pelo_blue": 1500.0, "pelo_red": 1500.0,
}
_GG_ROW = {
    "blue_team": "Alpha", "red_team": "Beta", "winner_side": "blue",
    "elo_blue_pre": 1500.0, "elo_red_pre": 1500.0,
    "elo_blue_pre_fast": 1500.0, "elo_red_pre_fast": 1500.0,
}


class TeamPriorTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(live.team_priors, "_cache"):
            del live.team_priors._cache

    def test_latest_pregame_rating_is_advanced_by_known_result(self):
        conn = _Conn([_OE_ROW], [_GG_ROW])
        out = live.team_priors(conn, ["Alpha", "Beta"])
        self.assertTrue(out["found"])
        self.assertAlmostEqual(out["elo_blue"], 1515.0)
        self.assertAlmostEqual(out["elo_red"], 1485.0)
        self.assertAlmostEqual(out["elo_oe"], 30.0 / 400.0)
        self.assertAlmostEqual(out["pelo_oe"], 24.0 / 400.0)
        self.assertEqual(out["form_diff"], 1.0)
        self.assertAlmostEqual(out["elo_gg"], 30.0 / 400.0)
        # the fast channel advances by its own K
        self.assertAlmostEqual(out["elo_gg_fast"], 120.0 / 400.0)

    def test_series_prior_is_oriented_blue_minus_red(self):
        self.assertEqual(live.series_prior({"wins": [1, 0]}), 1.0)
        self.assertEqual(live.series_prior({"wins": [0, 2]}), -2.0)
        self.assertEqual(live.series_prior({}), 0.0)

    def test_golgg_elo_carries_priors_while_oe_source_is_stale(self):
        conn = _Conn([], [_GG_ROW])
        out = live.team_priors(conn, ["Alpha", "Beta"])
        self.assertTrue(out["found"])
        self.assertFalse(out["oe_found"])
        self.assertEqual(out["elo_oe"], 0.0)
        self.assertEqual(out["form_diff"], 0.0)
        self.assertAlmostEqual(out["elo_gg"], 30.0 / 400.0)

    def test_cache_has_a_bounded_lifetime_shape(self):
        conn = _Conn([_OE_ROW], [_GG_ROW])
        live.team_priors(conn, ["Alpha", "Beta"])
        live.team_priors(conn, ["Alpha", "Beta"])
        self.assertEqual(conn.calls, 2)
        self.assertIn("at", live.team_priors._cache)

    def test_exact_rating_source_rosters_dates_and_availability_are_recorded(self):
        oe = dict(_OE_ROW, game_id="oe-source", date_utc=1900000000,
                  players_blue=["A1", "A2", "A3", "A4", "A5"],
                  players_red=["B1", "B2", "B3", "B4", "B5"])
        gg = dict(_GG_ROW, game_id=400, date=dt.datetime.fromtimestamp(1900086400, dt.timezone.utc))
        with mock.patch.object(live.time, "time", return_value=1900172800):
            out = live.team_priors(_Conn([oe], [gg]), ["Alpha", "Beta"])
        provenance = out["prior_provenance"]
        self.assertEqual(provenance["oe"]["blue"]["game_id"], "oe-source")
        self.assertEqual(provenance["oe"]["blue"]["players"], oe["players_blue"])
        self.assertEqual(provenance["oe"]["blue"]["age_days"], 2)
        self.assertEqual(provenance["golgg"]["blue"]["age_days"], 1)
        self.assertTrue(all(out["prior_availability"].values()))

    def test_missing_oe_prior_records_absence_instead_of_even_rating_evidence(self):
        out = live.team_priors(_Conn([], [_GG_ROW]), ["Alpha", "Beta"])
        self.assertFalse(out["prior_availability"]["elo_oe"])
        self.assertFalse(out["prior_provenance"]["oe"]["blue"]["available"])
        self.assertTrue(out["prior_availability"]["elo_gg"])


class FrameStateTests(unittest.TestCase):
    @staticmethod
    def _team(side, dead):
        participants = []
        for i in range(5):
            participants.append({
                "participantId": i + 1 + (5 if side == "red" else 0),
                "totalGold": 8000 + i * 100,
                "currentHealth": 0 if i < dead else 1000,
                "maxHealth": 1000,
                "level": 16,
                "creepScore": 200,
            })
        return {
            "participants": participants, "totalGold": 41000,
            "totalKills": 10, "towers": 5, "inhibitors": 0,
            "barons": 0, "dragons": [],
        }

    def test_missing_health_zeroes_hp_features_like_training(self):
        # The feed omits health in the first minutes; training marks those
        # states has_hp=0 with zeroed features, so live must not fabricate
        # full-health values.
        def team(side):
            t = self._team(side, dead=0)
            for p in t["participants"]:
                del p["currentHealth"], p["maxHealth"]
            return t
        frame = {"rfc460Timestamp": "2026-08-29T00:01:00Z",
                 "blueTeam": team("blue"), "redTeam": team("red")}
        state = live._frame_state({}, frame, None, None, {},
                                  live._ts("2026-08-29T00:00:00Z"))
        self.assertEqual(state["has_hp"], 0.0)
        self.assertEqual(state["hp_pool"], 0.0)
        self.assertEqual(state["hp_low_b"], 0.0)
        self.assertEqual(state["lvl_k"], 0.0)
        self.assertEqual(state["dead_blue"], 0)

    def test_legacy_prior_clip_covers_every_live_prior_channel(self):
        from lol_ticker import wpx
        for key in ("elo_oe", "pelo_oe", "form_diff", "elo_gg",
                    "elo_gg_fast", "series_diff"):
            self.assertIn(key, wpx.PRIOR_CLIP)

    def test_frame_deaths_come_from_window_health_not_details(self):
        frame = {
            "rfc460Timestamp": "2026-08-29T00:40:00Z",
            "blueTeam": self._team("blue", dead=4),
            "redTeam": self._team("red", dead=0),
        }
        # A real details participant has items, but no currentHealth.
        details = {"participants": [
            {"participantId": i, "items": [1001]} for i in range(1, 11)
        ]}
        state = live._frame_state(
            {}, frame, None, details, {1001: 300},
            live._ts("2026-08-29T00:00:00Z"),
        )
        self.assertEqual(state["dead_blue"], 4)
        self.assertEqual(state["dead_red"], 0)
        self.assertEqual(state["hp_low_b"], 4.0)

    def test_player_gold_has_absolute_blue_then_red_slots_and_explicit_missing_fallback(self):
        blue, red = self._team("blue", 0), self._team("red", 0)
        red["participants"][2]["totalGold"] = 12000
        frame = {"rfc460Timestamp": "2026-08-29T00:10:00Z", "blueTeam": blue, "redTeam": red}
        state = live._frame_state({}, frame, None, None, {}, live._ts("2026-08-29T00:00:00Z"))
        self.assertEqual(state["gold_players"], [8000, 8100, 8200, 8300, 8400, 8000, 8100, 12000, 8300, 8400])
        self.assertEqual(state["gold_role"][2], -3.8)
        red["participants"][2].pop("totalGold")
        self.assertIsNone(live._player_gold(blue, red))
        missing = live._frame_state({}, frame, None, None, {}, live._ts("2026-08-29T00:00:00Z"))
        self.assertIsNone(missing["gold_players"])
        self.assertIsNone(missing["gold_role"])

    def test_midgame_restart_recovers_recent_kill_clock(self):
        t0 = live._ts("2026-08-29T00:00:00Z")

        def frame(ts, kills):
            return {
                "rfc460Timestamp": live._iso(ts),
                "blueTeam": {"totalKills": kills, "dragons": [], "barons": 0},
                "redTeam": {"totalKills": 0, "dragons": [], "barons": 0},
            }

        current = frame(t0 + 500, 5)

        def probe(_game_id, ts):
            return frame(ts, 4 if ts < t0 + 400 else 5)

        with mock.patch.object(live, "_baron_probe", side_effect=probe):
            clock = live._seed_recent_kill_clock("g", current, 500, t0)
        self.assertIsNotNone(clock)
        self.assertLessEqual(abs(clock - 400), 10)

    def test_baron_timer_starts_at_180_and_active_means_timer_positive(self):
        frame = {
            "blueTeam": self._team("blue", dead=0),
            "redTeam": self._team("red", dead=0),
        }
        prev = {"baron_counts": [0, 0], "last_baron_clock": [None, None]}
        frame["redTeam"]["barons"] = 1
        self.assertEqual(live._update_baron_state(prev, frame, 100.0), [0.0, 180.0])
        self.assertEqual(live._update_baron_state(prev, frame, 279.0), [0.0, 1.0])
        self.assertEqual(live._update_baron_state(prev, frame, 280.0), [0.0, 0.0])

    def test_elder_timer_starts_at_150_and_elder_does_not_create_soul(self):
        frame = {
            "rfc460Timestamp": "2026-08-29T00:40:00Z",
            "blueTeam": self._team("blue", dead=0),
            "redTeam": self._team("red", dead=0),
        }
        frame["blueTeam"]["dragons"] = ["ocean", "cloud", "infernal", "elder"]
        prev = {"elder_counts": [0, 0], "last_elder_clock": [None, None]}
        self.assertEqual(live._update_elder_state(prev, frame, 100.0), [150.0, 0.0])
        self.assertEqual(live._update_elder_state(prev, frame, 249.0), [1.0, 0.0])
        self.assertEqual(live._update_elder_state(prev, frame, 250.0), [0.0, 0.0])
        state = live._frame_state(
            {}, frame, None, None, {}, live._ts("2026-08-29T00:00:00Z"))
        self.assertEqual(state["drag_blue"], 3)
        self.assertEqual(state["elders"], 1)
        self.assertFalse(state["soul_blue"])


class TeamNameTests(unittest.TestCase):
    def test_sponsor_aliases_fold_to_canonical_names(self):
        self.assertEqual(draft.norm_team("Cloud9 Kia"), "cloud9")
        self.assertEqual(draft.norm_team("Cloud9"), "cloud9")
        self.assertEqual(draft.norm_team("Team Liquid Alienware"), "liquid")
        # 2026-08-29 audit: feed name on the left, stored name on the right
        self.assertEqual(draft.norm_team("NONGSHIM RED FORCE"),
                         draft.norm_team("Nongshim RedForce"))
        self.assertEqual(draft.norm_team("Relove Deep Cross Gaming"),
                         draft.norm_team("Deep Cross Gaming"))
        self.assertEqual(draft.norm_team("Beijing JDG Esports"),
                         draft.norm_team("JD Gaming"))
        self.assertEqual(draft.norm_team("AG.AL"),
                         draft.norm_team("Anyone's Legend"))
        self.assertEqual(draft.norm_team("Saigon Warrior"),
                         draft.norm_team("Saigon Warriors"))
        self.assertEqual(draft.norm_team("TP.HCM SN CyberCore Esports"),
                         draft.norm_team("SN CyberCore Esports"))
        self.assertEqual(draft.norm_team("Brod & Friends"),
                         draft.norm_team("Brod n Friends"))
        self.assertEqual(draft.norm_team("KaBuM! Eports"),
                         draft.norm_team("KaBuM! Ilha das Lendas"))
        self.assertEqual(draft.norm_team("DK Challengers"),
                         draft.norm_team("Dplus KIA Challengers"))
        # 2026-09-03 shadow-ledger audit
        self.assertEqual(draft.norm_team("kt Challengers"),
                         draft.norm_team("KT Rolster Challengers"))
        self.assertEqual(draft.norm_team("Gamespace M.C."),
                         draft.norm_team("Gamespace MCE"))
        self.assertEqual(draft.norm_team("UP2U Meavedron"),
                         draft.norm_team("Meavedron"))

    def test_academy_rosters_stay_distinct(self):
        self.assertNotEqual(draft.norm_team("LYON Academy"),
                            draft.norm_team("LYON"))
        self.assertNotEqual(draft.norm_team("NONGSHIM RED FORCE"),
                            draft.norm_team("Nongshim RedForce Academy"))
        self.assertNotEqual(draft.norm_team("NS Challengers"),
                            draft.norm_team("Nongshim RedForce"))
        self.assertNotEqual(draft.norm_team("CTBC Flying Oyster Academy"),
                            draft.norm_team("CTBC Flying Oyster"))
        self.assertNotEqual(draft.norm_team("kt Challengers"),
                            draft.norm_team("KT Rolster"))


class _SeqConn:
    """Serves one prepared row set per execute() call, in order."""

    def __init__(self, results):
        self.results = list(results)

    def execute(self, _query, _params=None):
        self._current = self.results.pop(0)
        return self

    def fetchall(self):
        return self._current

    def __iter__(self):
        return iter(self._current)


class RosterAwarenessTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(live._reference_roster, "_cache"):
            del live._reference_roster._cache

    def test_tagged_summoner_names_match_reference_roster(self):
        matched, new, missing = live.match_lineup(
            ["Thanatos", "Oddie", "Saint", "Hena", "Lyonz"],
            ["LYON Thanatos", "LYON Oddie", "LYON Saint", "LYON Hena", "LYON Lyonz"])
        self.assertEqual((len(matched), new, missing), (5, [], []))

    def test_substitution_is_flagged(self):
        matched, new, missing = live.match_lineup(
            ["Blaber", "Berserker", "Jojopyun", "Thanatos", "VULCAN"],
            ["C9 Blaber", "C9 Berserker", "C9 Jojopyun", "C9 Thanatos", "C9 Zven"])
        self.assertEqual(new, ["C9 Zven"])
        self.assertEqual(missing, ["VULCAN"])

    def test_tagless_join_matches_by_suffix_but_short_names_do_not(self):
        matched, new, missing = live.match_lineup(["Faker"], ["T1Faker"])
        self.assertEqual((new, missing), ([], []))
        matched, new, missing = live.match_lineup(["Bo"], ["XYZ Rambo"])
        self.assertEqual(new, ["XYZ Rambo"])
        self.assertEqual(missing, ["Bo"])

    def test_reference_roster_skips_games_without_player_rows(self):
        games = [{"game_id": 9, "blue_team": "Alpha", "red_team": "Gamma",
                  "date": "2026-08-25"},
                 {"game_id": 7, "blue_team": "Alpha", "red_team": "Beta",
                  "date": "2026-08-20"}]
        alpha = [{"player": p} for p in ("A1", "A2", "A3", "A4", "A5")]
        conn = _SeqConn([games, [], alpha])   # newest game has a scrape gap
        ref = live._reference_roster(conn, "Alpha")
        self.assertEqual(ref["game_id"], 7)
        self.assertEqual(len(ref["players"]), 5)

    def test_roster_check_reports_per_side_changes(self):
        games = [{"game_id": 7, "blue_team": "Alpha", "red_team": "Beta",
                  "date": "2026-08-20"}]
        alpha = [{"player": p} for p in ("A1", "A2", "A3", "A4", "A5")]
        beta = [{"player": p} for p in ("B1", "B2", "B3", "B4", "B5")]
        conn = _SeqConn([games, alpha, beta])
        out = live.roster_check(
            conn, ["Alpha", "Beta"],
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertTrue(out["blue"]["changed"])
        self.assertEqual(out["blue"]["new"], ["ALP Sub9"])
        self.assertEqual(out["blue"]["missing"], ["A5"])
        self.assertEqual(out["blue"]["reference_game_id"], 7)
        self.assertFalse(out["red"]["changed"])
        self.assertEqual(out["red"]["matched"], 5)


class LineupAdjustedPriorTests(unittest.TestCase):
    def setUp(self):
        import time
        live._reference_roster._cache = {
            "at": time.time(), "teams": {
                "alpha": {"game_id": 7, "date": "2026-08-20",
                          "players": ["A1", "A2", "A3", "A4", "A5"]},
                "beta": {"game_id": 8, "date": "2026-08-20",
                         "players": ["B1", "B2", "B3", "B4", "B5"]},
            }, "games": []}
        table = {live._norm_player(p): (elo, 20, 100) for p, elo in
                 (("A1", 1500), ("A2", 1500), ("A3", 1500), ("A4", 1500),
                  ("A5", 1400), ("Sub9", 1800),
                  ("B1", 1500), ("B2", 1500), ("B3", 1500), ("B4", 1500), ("B5", 1500))}
        live._player_elos._cache = {"at": time.time(), "table": table}

    def tearDown(self):
        for fn in (live._reference_roster, live._player_elos):
            if hasattr(fn, "_cache"):
                del fn._cache

    def test_substitute_shifts_the_player_elo_prior(self):
        priors = {"found": True, "pelo_oe": (1480.0 - 1500.0) / 400.0,
                  "pelo_blue": 1480.0, "pelo_red": 1500.0}
        out = live.lineup_adjusted_priors(
            None, ["Alpha", "Beta"],
            priors,
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        # blue lineup mean with Sub9 (1800) = (4*1500+1800)/5 = 1560; red team pelo kept
        self.assertTrue(out["pelo_adjustment"]["blue"]["applied"])
        self.assertFalse(out["pelo_adjustment"]["red"]["applied"])
        self.assertAlmostEqual(out["pelo_oe"], (1560.0 - 1500.0) / 400.0)
        self.assertTrue(out["roster"]["blue"]["changed"])
        # the caller's dict is not mutated
        self.assertAlmostEqual(priors["pelo_oe"], -0.05)

    def test_unchanged_lineups_keep_the_advanced_team_pelo(self):
        priors = {"found": True, "pelo_oe": 0.1, "pelo_blue": 1540.0, "pelo_red": 1500.0}
        out = live.lineup_adjusted_priors(
            None, ["Alpha", "Beta"],
            priors,
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP A5"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertFalse(out["pelo_adjustment"]["blue"]["applied"])
        self.assertAlmostEqual(out["pelo_oe"], 0.1)

    def test_too_few_resolved_names_leave_the_prior_alone(self):
        priors = {"found": True, "pelo_oe": 0.0, "pelo_blue": 1500.0, "pelo_red": 1500.0}
        out = live.lineup_adjusted_priors(
            None, ["Alpha", "Beta"],
            priors,
            ["ALP X1", "ALP X2", "ALP X3", "ALP X4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertFalse(out["pelo_adjustment"]["blue"]["applied"])
        self.assertAlmostEqual(out["pelo_oe"], 0.0)

    def test_prior_less_teams_get_a_player_based_pelo(self):
        out = live.lineup_adjusted_priors(
            None, ["Alpha", "Beta"],
            {"found": False},
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertTrue(out["found"])
        self.assertAlmostEqual(out["pelo_oe"], (1560.0 - 1500.0) / 400.0)

    def test_oe_source_lineup_detects_substitution_hidden_by_newer_golgg_roster(self):
        # The newer gol.gg roster already includes Sub9. The actual player
        # prior still comes from OE, whose source game used A5.
        live._reference_roster._cache["teams"]["alpha"]["players"][-1] = "Sub9"
        source = {"oe": {
            "blue": {"available": True, "source": "oe_games+oe_ratings", "game_id": "old-oe", "date": 100,
                     "players": ["A1", "A2", "A3", "A4", "A5"], "age_days": 20},
            "red": {"available": True, "source": "oe_games+oe_ratings", "game_id": "old-red", "date": 100,
                    "players": ["B1", "B2", "B3", "B4", "B5"], "age_days": 20}}}
        out = live.lineup_adjusted_priors(None, ["Alpha", "Beta"],
            {"pelo_blue": 1480, "pelo_red": 1500, "prior_provenance": source},
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertTrue(out["pelo_adjustment"]["blue"]["applied"])
        self.assertEqual(out["roster"]["blue"]["reference_game_id"], "old-oe")
        self.assertEqual(out["roster"]["blue"]["reference_source"], "oe_games+oe_ratings")
        resolved = out["pelo_adjustment"]["blue"]["resolved_players"]
        self.assertEqual(len(resolved), 5)
        self.assertEqual(resolved[-1]["matched_name"], "sub9")
        self.assertEqual(resolved[-1]["games"], 20)
        self.assertIn("age_days", resolved[-1])

    def test_missing_oe_lineup_does_not_fabricate_reference_from_golgg(self):
        out = live.lineup_adjusted_priors(None, ["Alpha", "Beta"],
            {"pelo_blue": 1500, "pelo_red": 1500, "pelo_oe": 0, "prior_provenance": {"oe": {}}},
            ["ALP A1", "ALP A2", "ALP A3", "ALP A4", "ALP Sub9"],
            ["BET B1", "BET B2", "BET B3", "BET B4", "BET B5"])
        self.assertFalse(out["roster"]["blue"]["available"])
        self.assertFalse(out["pelo_adjustment"]["blue"]["applied"])
        self.assertEqual(out["pelo_adjustment"]["blue"]["reason"], "rating_source_lineup_unavailable")


class FeedSequenceTests(unittest.TestCase):
    def setUp(self):
        self.original_cache = live._cache
        live._cache = {}
        self.t0 = live._ts("2026-08-29T00:00:00Z")
        self.md = {side+"TeamMetadata": {"esportsTeamId": side, "participantMetadata": [
            {"participantId": i+1+(5 if side == "red" else 0), "championId": "Ahri", "summonerName": side+str(i)}
            for i in range(5)]} for side in ("blue", "red")}
        conn = mock.MagicMock()
        conn.execute.return_value = []
        self.cache = live._new_cache(conn, self.md, self.t0)
        self.cache["prev"].update(kill_seeded=True, kills=20, baron_seeded=True, elder_seeded=True,
                                  baron_counts=[0, 0], elder_counts=[0, 0])
        live._cache["g"] = self.cache

    def tearDown(self):
        live._cache = self.original_cache

    def frame(self, second, state="in_game", event=False):
        blue, red = FrameStateTests._team("blue", 0), FrameStateTests._team("red", 0)
        if event:
            blue.update(barons=1, dragons=["elder"], totalKills=11)
        return {"rfc460Timestamp": live._iso(self.t0+second), "gameState": state,
                "blueTeam": blue, "redTeam": red}

    def test_missing_fresh_frames_still_returns_player_identities(self):
        self.md["blueTeamMetadata"]["participantMetadata"][0]["role"] = "top"
        with mock.patch.object(live, "_get", return_value={"frames": []}):
            result = live.estimate_series(None, "g")
        self.assertEqual(result["frames"], [])
        self.assertEqual(len(result["lineup"]), 10)
        self.assertEqual(result["lineup"][0], {
            "side": "blue", "pid": 1, "player": "blue0", "role": "top", "champion": "Ahri"})
        self.assertNotIn("scoreboard", result)

    def estimate(self, frames, priors=None):
        from lol_ticker import wpx
        forecast = {"p_blue": .6, "lo_prior": 0, "lo_state": 0, "lo_champ": 0, "lo_time": 0}
        window = {"gameMetadata": self.md, "frames": frames}
        with mock.patch.object(live, "_get", side_effect=[window, {"frames": []}, {"frames": []}]), \
                mock.patch.object(wpx, "predict_live", return_value=forecast):
            return live.estimate_series(None, "g", priors=priors)

    def test_repeated_overlapping_paused_windows_preserve_clock_objectives_and_kills(self):
        first = [self.frame(120), self.frame(121, "paused", True), self.frame(122, "paused", True)]
        result1 = self.estimate(first)
        original = copy.deepcopy(self.cache["prev"])
        repeated = self.estimate(first)
        self.assertEqual(self.cache["prev"], original)
        self.assertEqual(result1["frames"], repeated["frames"])
        overlap = self.estimate(first[1:]+[self.frame(123, "in_game", True), self.frame(124, "in_game", True)])
        self.assertEqual(overlap["frames"][:2], result1["frames"][1:])
        self.assertEqual(self.cache["prev"]["pause_s"], 2)
        latest = overlap["frames"][-1]
        self.assertEqual(latest["clock_s"], 122)
        self.assertEqual(latest["baron_timer_blue_s"], 179)
        self.assertEqual(latest["elder_timer_blue_s"], 149)
        self.assertAlmostEqual(latest["t_since_kill_min"], 1/60)
        # Caller mutation cannot change the cached forecast on a retry.
        overlap["frames"][-1]["p_blue"] = 0
        again = self.estimate([self.frame(124, "in_game", True)])
        self.assertEqual(again["frames"][0]["p_blue"], .6)

    def test_out_of_order_window_is_processed_in_timestamp_order(self):
        result = self.estimate([self.frame(122, event=True), self.frame(120), self.frame(121, event=True)])
        self.assertEqual([r["ts"] for r in result["frames"]], [self.t0+120, self.t0+121, self.t0+122])
        self.assertEqual(result["frames"][-1]["baron_timer_blue_s"], 179)

    def test_reused_frames_accept_corrected_side_priors_without_replaying_pause(self):
        frames = [self.frame(120, "paused"), self.frame(121, "in_game", True)]
        first = self.estimate(frames, priors={"elo_oe": -.5, "series_diff": -1})
        original = copy.deepcopy(self.cache["prev"])
        corrected = self.estimate(frames, priors={"elo_oe": .5, "series_diff": 1})
        self.assertEqual(first["frames"][-1]["elo_oe"], -.5)
        self.assertEqual(corrected["frames"][-1]["elo_oe"], .5)
        self.assertEqual(corrected["frames"][-1]["series_diff"], 1)
        self.assertEqual(self.cache["prev"], original)

    def test_all_missing_feed_windows_share_one_deadline_and_release_it(self):
        clock = [0.0]
        requests = []
        def slow_empty(*args, **kwargs):
            requests.append(live._request_timeout(30))
            clock[0] += 9
            live._request_timeout(30)
            return None
        with mock.patch.object(live.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(live, "_get", side_effect=slow_empty):
            with self.assertRaisesRegex(TimeoutError, "budget exhausted"):
                live.estimate_series(None, "g", request_budget_s=20)
        self.assertEqual(requests, [20, 11, 2])
        self.assertIsNone(live._REQUEST_DEADLINE.get())


class MarketResolutionTests(unittest.TestCase):
    def test_red_kalunga_resolves_game_four_under_exchange_name(self):
        polymarket = [{"markets": [{
            "question": "LoL: paiN Gaming vs RED Canids - Game 4 Winner",
            "outcomes": '["paiN Gaming", "RED Canids"]',
            "clobTokenIds": '["pain-token", "red-token"]',
            "outcomePrices": '["0.4", "0.6"]',
            "closed": False, "acceptingOrders": True,
        }]}]
        with mock.patch.object(live, "_kalshi_markets", return_value=[]), \
                mock.patch.object(live, "_gamma_events", return_value=polymarket):
            for name in ("RED Kalunga", "RED Canids Kalunga"):
                resolved = live.resolve_markets([name, "paiN Gaming"], 4)
                self.assertEqual(resolved["pm"], {
                    "pain": "pain-token", "red canids": "red-token"})
                self.assertEqual(resolved["pm_src"], "map")
            self.assertFalse(live.resolve_markets(
                ["RED Kalunga Academy", "paiN Gaming"], 4)["pm"])
            self.assertFalse(live.resolve_markets(
                ["RED Kalunga", "paiN Gaming"], 3)["pm"])

    def test_sponsor_alias_and_terminal_map_title_resolve_both_exchanges(self):
        self.assertEqual(draft.norm_team("Team Liquid Alienware"), "liquid")
        kalshi = [
            {"ticker": "TL-M1", "title": "Team Liquid wins map 1",
             "yes_sub_title": "Team Liquid"},
            {"ticker": "SR-M1", "title": "Shopify Rebellion wins map 1",
             "yes_sub_title": "Shopify Rebellion"},
        ]
        polymarket = [{"markets": [{
            "question": "LoL: Team Liquid vs Shopify Rebellion - Game 1 Winner",
            "outcomes": '["Team Liquid", "Shopify Rebellion"]',
            "clobTokenIds": '["tl-token", "sr-token"]',
            "outcomePrices": '["0.6", "0.4"]',
            "slug": "lol-tl2-sr-game1", "closed": False,
            "acceptingOrders": True,
        }]}]
        with mock.patch.object(live, "_kalshi_markets", return_value=kalshi), \
                mock.patch.object(live, "_gamma_events", return_value=polymarket):
            resolved = live.resolve_markets(
                ["Team Liquid Alienware", "Shopify Rebellion"], 1)
        self.assertEqual(resolved["kalshi"], {
            "liquid": "TL-M1", "shopify rebellion": "SR-M1"})
        self.assertEqual(resolved["pm"], {
            "liquid": "tl-token", "shopify rebellion": "sr-token"})
        self.assertTrue(resolved["complete"])


class MarketEventMatchingTests(unittest.TestCase):
    """Kalshi markets are matched by event holding both teams, nearest in time."""

    def _kalshi(self):
        return [
            {"ticker": "KXLOLMAP-26SEP031000BRTNBS-1-BRT", "event_ticker": "KXLOLMAP-26SEP031000BRTNBS-1",
             "title": "BRUTE wins map 1", "yes_sub_title": "BRUTE"},
            {"ticker": "KXLOLMAP-26SEP031000BRTNBS-1-NBS", "event_ticker": "KXLOLMAP-26SEP031000BRTNBS-1",
             "title": "NightBirds wins map 1", "yes_sub_title": "NightBirds"},
            {"ticker": "KXLOLMAP-26SEP011230BRTMEA-1-BRT", "event_ticker": "KXLOLMAP-26SEP011230BRTMEA-1",
             "title": "BRUTE wins map 1", "yes_sub_title": "BRUTE"},
            {"ticker": "KXLOLMAP-26SEP011230BRTMEA-1-MEA", "event_ticker": "KXLOLMAP-26SEP011230BRTMEA-1",
             "title": "Meavedron wins map 1", "yes_sub_title": "Meavedron"},
        ]

    def test_ticker_time_is_us_eastern(self):
        t = live.ticker_time("KXLOLMAP-26SEP031000BRTNBS-1-BRT")
        self.assertEqual(t, 1788444000.0)   # 2026-09-03 10:00 EDT = 14:00Z
        self.assertIsNone(live.ticker_time("TL-M1"))

    def test_requires_both_teams_in_one_event_and_prefers_nearest(self):
        # Sep 1 game: the Sep 3 NightBirds event also lists BRUTE but not the opponent.
        start = live.ticker_time("KXLOLMAP-26SEP011230BRTMEA-1-BRT") + 3600
        with mock.patch.object(live, "_kalshi_markets", return_value=self._kalshi()), \
                mock.patch.object(live, "_gamma_events", return_value=[]):
            resolved = live.resolve_markets(["BRUTE", "UP2U Meavedron"], 1, game_start_ts=start)
        self.assertEqual(resolved["kalshi"], {
            "brute": "KXLOLMAP-26SEP011230BRTMEA-1-BRT",
            draft.norm_team("UP2U Meavedron"): "KXLOLMAP-26SEP011230BRTMEA-1-MEA"})
        self.assertEqual(resolved["kalshi_event"], "KXLOLMAP-26SEP011230BRTMEA-1")

    def test_single_team_match_no_longer_quotes_another_series(self):
        with mock.patch.object(live, "_kalshi_markets", return_value=self._kalshi()), \
                mock.patch.object(live, "_gamma_events", return_value=[]):
            resolved = live.resolve_markets(["BRUTE", "Some Other Team"], 1)
        self.assertEqual(resolved["kalshi"], {})

    def test_event_too_far_from_game_start_is_rejected(self):
        start = live.ticker_time("KXLOLMAP-26SEP031000BRTNBS-1-BRT") + 3 * 86400
        with mock.patch.object(live, "_kalshi_markets", return_value=self._kalshi()), \
                mock.patch.object(live, "_gamma_events", return_value=[]):
            resolved = live.resolve_markets(["BRUTE", "NightBirds"], 1, game_start_ts=start)
        self.assertEqual(resolved["kalshi"], {})
        self.assertFalse(live.ticker_offset_ok("KXLOLMAP-26SEP031000BRTNBS-1-BRT", start))
        self.assertTrue(live.ticker_offset_ok("KXLOLMAP-26SEP031000BRTNBS-1-BRT", start - 3 * 86400 + 86400))

    def test_known_opponent_alias_matches_a_two_team_event(self):
        kalshi = self._kalshi() + [
            {"ticker": "KXLOLMAP-26SEP031330DV1LDS-2-DV1", "event_ticker": "KXLOLMAP-26SEP031330DV1LDS-2",
             "title": "devils.one x KMT wins map 2", "yes_sub_title": "devils.one x KMT"},
            {"ticker": "KXLOLMAP-26SEP031330DV1LDS-2-LDS", "event_ticker": "KXLOLMAP-26SEP031330DV1LDS-2",
             "title": "LODIS wins map 2", "yes_sub_title": "LODIS"},
        ]
        start = live.ticker_time("KXLOLMAP-26SEP031330DV1LDS-2-DV1") + 2400
        with mock.patch.object(live, "_kalshi_markets", return_value=kalshi), \
                mock.patch.object(live, "_gamma_events", return_value=[]):
            resolved = live.resolve_markets(["LODIS", "DV1 inStreamly"], 2, game_start_ts=start)
        self.assertEqual(resolved["kalshi"], {
            "lodis": "KXLOLMAP-26SEP031330DV1LDS-2-LDS",
            "dv1 instreamly": "KXLOLMAP-26SEP031330DV1LDS-2-DV1"})

    def test_sole_event_cannot_infer_an_unknown_opponent(self):
        kalshi = self._kalshi()[:2]  # BRUTE vs NightBirds is the only listed event.
        start = live.ticker_time(kalshi[0]["ticker"])
        for source in ("KXLOLMAP", "KXLOLGAME"):
            with self.subTest(source=source), \
                    mock.patch.object(live, "_kalshi_markets",
                                      side_effect=lambda series: kalshi if series == source else []), \
                    mock.patch.object(live, "_gamma_events", return_value=[]):
                resolved = live.resolve_markets(
                    ["BRUTE", "Some Other Team"], 1, deciding=True, game_start_ts=start)
            self.assertEqual(resolved["kalshi"], {})

    def test_tail_match_never_equates_distinct_orgs(self):
        self.assertEqual(live.team_match("karmine corp", "karmine corp blue"), 0)
        self.assertEqual(live.team_match("meavedron", "up2u meavedron"), 1)
        self.assertEqual(live.team_match("brute", "brute"), 2)


if __name__ == "__main__":
    unittest.main()

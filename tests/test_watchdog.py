import unittest

from lol_ticker import watchdog as wd


class FindProblemsTests(unittest.TestCase):
    def test_healthy_idle_and_live(self):
        up = {"record": True, "shadow": True}
        self.assertEqual(wd.find_problems(up, 3600, 0), {})
        self.assertEqual(wd.find_problems(up, 60, 2), {})

    def test_dead_daemon_reported_unknown_is_not(self):
        got = wd.find_problems({"record": False, "shadow": None}, 60, 0)
        self.assertEqual(set(got), {"daemon:record"})

    def test_stale_books_threshold_depends_on_live_games(self):
        up = {"record": True}
        age = wd.LIVE_STALE_S + 60
        self.assertIn("books", wd.find_problems(up, age, 1))
        self.assertEqual(wd.find_problems(up, age, 0), {})
        # schedule unreachable: only the idle threshold applies
        self.assertEqual(wd.find_problems(up, age, None), {})
        self.assertIn("books", wd.find_problems(up, wd.IDLE_STALE_S + 1, None))
        self.assertIn("books", wd.find_problems(up, None, 0))


class PlanAlertsTests(unittest.TestCase):
    def test_new_problem_alerts_then_throttles_then_repeats(self):
        problems = {"daemon:record": "record daemon is not running"}
        msgs, state = wd.plan_alerts(problems, {}, now=1000)
        self.assertEqual(msgs, ["record daemon is not running"])
        msgs, state2 = wd.plan_alerts(problems, state, now=1000 + 300)
        self.assertEqual(msgs, [])
        self.assertEqual(state2, state)
        msgs, state3 = wd.plan_alerts(problems, state2, now=1000 + wd.REPEAT_S)
        self.assertEqual(msgs, ["record daemon is not running"])
        self.assertEqual(state3["daemon:record"]["since"], 1000)

    def test_recovery_is_announced_once(self):
        state = {"books": {"since": 0, "last_alert": 0}}
        msgs, state = wd.plan_alerts({}, state, now=1800)
        self.assertEqual(msgs, ["recovered: books (after 30 min)"])
        self.assertEqual(state, {})
        self.assertEqual(wd.plan_alerts({}, state, now=2000)[0], [])


if __name__ == "__main__":
    unittest.main()

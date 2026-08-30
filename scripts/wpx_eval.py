"""Re-evaluate the shipped odds-free models (5-fold by game) + fair minute-mark
comparison vs the markets, and write data/wpx/results.json for the dashboard.
Usage: python3 scripts/wpx_eval.py   (~4-6 min)"""
import json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
from lol_ticker import db, wpbench, wpblend, wpdeploy, wpgam, wpx, wpx_fair

ONLY = ["champscale_reg(l2=800,cap15)", "champscale_reg(l2=200,cap25)", "logit_rich_v3_xt"]
FAIR = ["champscale_reg(l2=800,cap15)"]
results = wpx.evaluate(only=ONLY)
wpx.report(results)
gam = wpgam.evaluate_walk_forward(bootstrap=1000)
wpgam.report_walk_forward(gam)
rolling = wpgam.evaluate_rolling(windows=2, bootstrap=500)
bench = wpbench.run(bootstrap=1000)
wpbench.report(bench)
blend = wpblend.run(bootstrap=1000)
wpblend.report(blend)
conn = db.connect()
fair = {}
for name in FAIR:
    for lag in (0, -45):
        oof_name = name.split("(")[0].replace("+", "_")
        r = wpx_fair.run(conn, oof_name=oof_name, lag_s=lag)
        fair["%s|lag%d" % (name, lag)] = r
        print("FAIR %s lag=%d" % (name, lag))
        for plat, v in r.items():
            print("   %s %s" % (plat, json.dumps(v)))
blend_latency45 = wpblend.run_latency(conn, lag_s=45, bootstrap=1000)
wpblend.report(blend_latency45)
rows = [{k: r.get(k) for k in ("model", "all", "pm", "ks", "pm_ll", "ks_ll", "pm_early", "pm_mid", "pm_late")} for r in results]
mk = next((r for r in results if r.get("mkt_pm_early") is not None), None)
mbp = {"early": mk["mkt_pm_early"], "mid": mk["mkt_pm_mid"], "late": mk["mkt_pm_late"]} if mk else None
out = os.path.join(wpx.OUT_DIR, "results.json")
json.dump(gam, open(os.path.join(wpx.OUT_DIR, "walk_forward_diagnostics.json"), "w"), indent=1)
json.dump(rolling, open(os.path.join(wpx.OUT_DIR, "rolling_origin.json"), "w"), indent=1)
stack_gate = (json.load(open(wpdeploy.RESULT_PATH))
              if os.path.exists(wpdeploy.RESULT_PATH) else None)
json.dump({"scoreboard": rows, "fair": fair, "market_by_phase": mbp,
           "constrained_walk_forward": gam, "method_benchmark": bench,
           "rolling_origin": rolling,
           "blend_benchmark": blend,
           "blend_latency45": blend_latency45,
           "live_stack_benchmark": stack_gate}, open(out, "w"), indent=1)
print("wrote", out)

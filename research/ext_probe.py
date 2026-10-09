"""Probe the LoL Esports Overlay extension backend (socket.io) with a Twitch
extension JWT and log every event for a while, so the schema can be mapped.

Usage:
  python3 research/ext_probe.py --jwt <JWT> [--channel <twitch_channel_id>] [--url URL]
                               [--path socket.io] [--namespace /] [--seconds 60]
                               [--auth query|auth|header|all] [--emit EVENT[:JSON]]
Writes a transcript to data/ext_probe.log (and prints it).
"""
import argparse
import json
import time

import socketio

ap = argparse.ArgumentParser()
ap.add_argument("--jwt", required=True)
ap.add_argument("--channel", default=None, help="twitch channel id (numeric) if the server wants it")
ap.add_argument("--url", default="https://lolesports-extension-backend.fracassi.tech")
ap.add_argument("--path", default="socket.io")
ap.add_argument("--namespace", default="/")
ap.add_argument("--seconds", type=int, default=60)
ap.add_argument("--auth", default="all", choices=["query", "auth", "header", "all"])
ap.add_argument("--emit", action="append", default=[],
                help="event to emit after connect, e.g. 'join:{\"channelId\":\"123\"}'")
a = ap.parse_args()

log_f = open("data/ext_probe.log", "a")


def log(*x):
    line = "%s %s" % (time.strftime("%H:%M:%S"), " ".join(str(i) for i in x))
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()


def attempt(mode):
    sio = socketio.Client(logger=False, engineio_logger=False, reconnection=False)
    got = {"events": 0}

    @sio.on("connect", namespace=a.namespace)
    def on_connect():
        log("[%s] connected; sid=%s" % (mode, sio.sid))
        for e in a.emit:
            name, _, payload = e.partition(":")
            data = json.loads(payload) if payload else None
            log("[%s] emit %s %s" % (mode, name, data))
            sio.emit(name, data, namespace=a.namespace)

    @sio.on("connect_error", namespace=a.namespace)
    def on_err(d):
        log("[%s] connect_error: %s" % (mode, str(d)[:300]))

    @sio.on("disconnect", namespace=a.namespace)
    def on_dc():
        log("[%s] disconnected" % mode)

    @sio.on("*", namespace=a.namespace)
    def catch_all(event, *args):
        got["events"] += 1
        s = json.dumps(args, default=str)
        log("[%s] EVENT %s (%d bytes): %s" % (mode, event, len(s), s[:1500]))

    url = a.url
    kw = {"socketio_path": a.path, "namespaces": [a.namespace],
          "transports": ["websocket", "polling"], "wait_timeout": 15}
    q = {"token": a.jwt}
    if a.channel:
        q["channelId"] = a.channel
    if mode == "query":
        url = a.url + "?" + "&".join("%s=%s" % kv for kv in q.items())
    elif mode == "auth":
        kw["auth"] = {"token": a.jwt, **({"channelId": a.channel} if a.channel else {})}
    elif mode == "header":
        kw["headers"] = {"Authorization": "Bearer " + a.jwt}
    try:
        log("[%s] connecting %s" % (mode, a.url))
        sio.connect(url, **kw)
    except Exception as e:
        log("[%s] FAILED: %s" % (mode, str(e)[:300]))
        return False
    t_end = time.time() + a.seconds
    while time.time() < t_end and sio.connected:
        time.sleep(1)
    log("[%s] done: %d events" % (mode, got["events"]))
    try:
        sio.disconnect()
    except Exception:
        pass
    return True


modes = ["query", "auth", "header"] if a.auth == "all" else [a.auth]
for m in modes:
    if attempt(m):
        break

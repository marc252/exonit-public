#!/usr/bin/env python3
"""Local-only ExoNit fake server for Swift integration tests."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--fits", required=True)
parser.add_argument("--state", required=True)
parser.add_argument("--port", type=int, default=0)
args = parser.parse_args()
fits_data = Path(args.fits).read_bytes()
state_path = Path(args.state)
state = {"task_polls": 0, "heartbeats": 0, "results": []}
lock = threading.Lock()

def save():
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n")

class Handler(BaseHTTPRequestHandler):
    server_version = "ExoNitFake/1"
    def log_message(self, format, *args): pass
    def json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")
    def send_json(self, status, value):
        data = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def authorized(self): return self.headers.get("Authorization") == "Bearer fake-token"
    def do_POST(self):
        if not self.authorized(): return self.send_json(401, {"error":"unauthorized"})
        body = self.json_body()
        if self.path == "/api/worker/register": return self.send_json(200, {"worker_id":"fake-wid"})
        if self.path == "/api/worker/heartbeat":
            with lock: state["heartbeats"] += 1; save()
            return self.send_json(200, {"command":"continue"})
        if self.path == "/api/worker/result":
            with lock: state["results"].append(body); save()
            return self.send_json(200, {"status":"ok"})
        if self.path in ("/api/worker/fpp_result", "/api/worker/exominer_result"):
            return self.send_json(200, {"status":"ok"})
        return self.send_json(404, {"error":"not_found"})
    def do_GET(self):
        if not self.authorized(): return self.send_json(401, {"error":"unauthorized"})
        if self.path == "/api/worker/task":
            with lock:
                state["task_polls"] += 1; poll = state["task_polls"]; save()
            return self.send_json(200, {
                "task_id":"fake-task", "tic_id":"123",
                "config":{"n_periods":600,"max_planets":1,"run_injection_recovery":False}
            } if poll == 1 else {"task_id":None,"wait_seconds":0.1})
        if self.path == "/api/metadata/123":
            return self.send_json(200, {"tic_id":"123","fits_files":["fixture.fits"],"sectors":[1],"ra":1.0,"dec":2.0,"tmag":10.0,"stellar_radius":1.0,"stellar_mass":1.0})
        if self.path == "/api/fits/fixture.fits":
            self.send_response(200); self.send_header("Content-Type", "application/fits")
            self.send_header("Content-Length", str(len(fits_data))); self.end_headers(); self.wfile.write(fits_data); return
        if self.path in ("/api/worker/fpp_task", "/api/worker/exominer_task"):
            return self.send_json(200, {"task":None})
        if self.path == "/api/worker/update": return self.send_json(200, {"files":[]})
        return self.send_json(404, {"error":"not_found"})

server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
save()
print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}), flush=True)
try: server.serve_forever()
except KeyboardInterrupt: pass
finally: server.server_close()

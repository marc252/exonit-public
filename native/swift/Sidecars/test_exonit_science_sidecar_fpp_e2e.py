#!/usr/bin/env python3
"""Loopback E2E for the science sidecar FPP mode."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PYTHON = Path(os.environ.get(
    "EXONIT_REFERENCE_PYTHON", "~/exonit-worker/venv/bin/python"
))
ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "Sidecars" / "exonit_science_sidecar.py"
WORKER_PYTHON_DIR = ROOT / "reference-python"


class FakeNearbyHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str | None]] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.__class__.requests.append((self.path, self.headers.get("Authorization")))
        if self.path != "/api/nearby/123":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(
            {
                "source": "local-fake",
                "gal_b": 30.0,
                "stars": [
                    {
                        "ID": "123",
                        "ra": 10.0,
                        "dec": 20.0,
                        "Tmag": 10.0,
                        "mass": 1.0,
                        "rad": 1.0,
                        "Teff": 5777.0,
                        "plx": 10.0,
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_fpp_sidecar_against_loopback_fake() -> None:
    assert PYTHON.is_file(), PYTHON
    FakeNearbyHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeNearbyHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    assert url.startswith("http://127.0.0.1:")

    try:
        time = [100.0 + i * 0.1 for i in range(80)]
        flux = [0.99 if abs(((t - 100.0) % 2.0) - 1.0) < 0.1 else 1.0 for t in time]
        payload = {
            "protocol": 1,
            "mode": "fpp",
            "worker_python_dir": str(WORKER_PYTHON_DIR),
            "workdir": str(ROOT / ".tmp-fpp-e2e"),
            "server_url": url,
            "token": "fake-token",
            "sector_count": 1,
            "fpp_n": 8,
            "time": time,
            "flux": flux,
            "task": {
                "tic_id": "123",
                "period_days": 2.0,
                "t0": 100.0,
                "duration_days": 0.2,
                "depth_ppm": 10000.0,
                "n_sectors": 1,
            },
        }
        completed = subprocess.run(
            [str(PYTHON), str(SIDECAR)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
            cwd=ROOT,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["protocol"] == 1
        assert result["status"] == "ok"
        assert 0.0 <= result["fpp"] <= 1.0
        assert 0.0 <= result["nfpp"] <= 1.0
        assert result["verdict"] in {"validated", "candidate", "likely_fp"}
        assert FakeNearbyHandler.requests == [
            ("/api/nearby/123", "Bearer fake-token")
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", server.server_port)) != 0


if __name__ == "__main__":
    test_fpp_sidecar_against_loopback_fake()

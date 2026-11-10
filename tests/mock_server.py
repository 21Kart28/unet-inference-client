"""A stand-in MONAI Label server for tests and for running the app without MONAI.

Run it by hand with:  python tests/mock_server.py --port 8000
"""

from __future__ import annotations

import argparse
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

NRRD_HEADER = "NRRD0004\ntype: uint8\ndimension: 3\nsizes: 4 4 4\nspacings: 1 1 1\nencoding: raw\n\n"
PREDICTION = NRRD_HEADER.encode() + bytes([0] * 21 + [1] * 22 + [0] * 21)


class MockMonaiLabel:
    def __init__(self, models=("deepedit_seg",), cases=None, device="cuda:0", gzip_output=False):
        self.models = list(models)
        self.cases = dict(cases or {"CASE_001": True, "CASE_002": False, "CASE_003": True})
        self.device = device
        self.gzip_output = gzip_output
        self.infer_response = PREDICTION
        self.infer_calls = []
        self.url = ""
        self._server = None

    def start(self, port: int = 0) -> str:
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/info/":
                    models = {name: {"config": {"device": [mock.device]}} for name in mock.models}
                    self._json({"models": models})
                elif path == "/datastore/":
                    objects = {
                        case: {"image": {"ext": ".nrrd"},
                               "labels": {"final": {"ext": ".nrrd"}} if has_label else {}}
                        for case, has_label in mock.cases.items()
                    }
                    self._json({"objects": objects})
                else:
                    self.send_error(404)

            def do_POST(self):
                parsed = urlparse(self.path)
                model = parsed.path.removeprefix("/infer/")
                image = parse_qs(parsed.query).get("image", [""])[0]
                mock.infer_calls.append((model, image))
                if model not in mock.models or image not in mock.cases:
                    self.send_error(404)
                    return
                body = mock.infer_response
                if mock.gzip_output:
                    body = gzip.compress(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve fake MONAI Label endpoints.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cases", type=int, default=5, help="number of fake cases")
    parser.add_argument("--gzip", action="store_true", help="gzip the prediction, as the real server does")
    args = parser.parse_args()
    cases = {f"CASE_{n:03d}": n % 3 != 0 for n in range(1, args.cases + 1)}
    mock = MockMonaiLabel(cases=cases, gzip_output=args.gzip)
    print(f"Mock MONAI Label server on {mock.start(args.port)}  (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        mock.stop()

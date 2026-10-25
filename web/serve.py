#!/usr/bin/env python3
"""Static file server for the demo: correct MIME types, optional cross-origin isolation headers (--isolate, for multi-threaded WASM),
HTTP Range support (so the 1.2 GB weights can be fetched/resumed efficiently). Usage:  python3 serve.py [port] [--isolate]"""
import http.server
import mimetypes
import os
import re
import sys

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/octet-stream", ".data")


class Handler(http.server.SimpleHTTPRequestHandler):
    isolate = False  # --isolate adds COOP/COEP (multi-threaded WASM); leave off when loading models from another origin (e.g. Hugging Face)

    def end_headers(self):
        if self.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if rng and os.path.isfile(path):
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            size = os.path.getsize(path)
            start = int(m.group(1)) if m and m.group(1) else 0
            end = int(m.group(2)) if m and m.group(2) else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        if "/models/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Handler.isolate = "--isolate" in sys.argv
    port = int(args[0]) if args else 8000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"serving {os.getcwd()} at http://localhost:{port}/   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass

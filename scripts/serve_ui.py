"""Static server for the UI with browser caching disabled.

`python -m http.server` sends Last-Modified but no Cache-Control, so
browsers may silently reuse stale copies of index.html/app.js after code
changes (heuristic caching). This wrapper serves ui/ with
`Cache-Control: no-store`, so every (re)load fetches the current files.

    python scripts/serve_ui.py [port]     # default port 3000
"""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
UI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        # /__version: latest mtime of the ui/ files. The page polls this
        # and reloads itself when it changes (dev auto-reload in app.js).
        if self.path == "/__version":
            stamp = str(max(
                os.path.getmtime(os.path.join(UI_DIR, f))
                for f in os.listdir(UI_DIR)
                if os.path.isfile(os.path.join(UI_DIR, f)))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(stamp)))
            self.end_headers()
            self.wfile.write(stamp)
            return
        super().do_GET()


if __name__ == "__main__":
    print(f"Serving {UI_DIR} on http://localhost:{PORT} (Cache-Control: no-store)")
    http.server.ThreadingHTTPServer(("", PORT), NoCacheHandler).serve_forever()

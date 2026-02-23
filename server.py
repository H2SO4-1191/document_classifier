import os
import io
import json
import webbrowser
import threading
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from ultralytics import YOLO
from PIL import Image

PORT       = 8000
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models") 
HTML_FILE  = "document-classifier.html"

MODELS = {
    "v1": os.path.join(MODELS_DIR, "document_classifier_v1.pt"),
    "v2": os.path.join(MODELS_DIR, "document_classifier_v2.pt"),
    "v3": os.path.join(MODELS_DIR, "document_classifier_v3.pt"),
}

# Load models at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DocScan] Device: {device}")

loaded_models = {}
for version, path in MODELS.items():
    if os.path.exists(path):
        print(f"[DocScan] Loading model {version} ...")
        m = YOLO(path)
        m.to(device)
        loaded_models[version] = m
        print(f"[DocScan] Model {version} ready")
    else:
        print(f"[DocScan] Model {version} not found at {path} (skipping)")

# Multipart parser
def parse_multipart(rfile, content_type_header, content_length):
    """
    Minimal multipart/form-data parser.
    Returns a dict of { field_name: bytes }.
    Works on Python 3.13+ where the cgi module was removed.
    """
    boundary = None
    for segment in content_type_header.split(";"):
        segment = segment.strip()
        if segment.startswith("boundary="):
            boundary = segment[len("boundary="):].strip().encode()
            break
    if not boundary:
        raise ValueError("No boundary in Content-Type header")

    body = rfile.read(content_length)
    fields = {}
    delimiter = b"--" + boundary

    for chunk in body.split(delimiter):
        if b"\r\n\r\n" not in chunk:
            continue
        headers_raw, _, data = chunk.partition(b"\r\n\r\n")
        headers_raw = headers_raw.strip(b"\r\n")
        if not headers_raw or headers_raw == b"--":
            continue
        data = data.rstrip(b"\r\n")

        name = None
        for line in headers_raw.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition"):
                for part in line.split(b";"):
                    part = part.strip()
                    if part.startswith(b'name="'):
                        name = part[6:-1].decode()
        if name:
            fields[name] = data

    return fields

# Classify helper
def classify(model, image_bytes):
    img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(img)
    probs   = results[0].probs.data.tolist()
    return float(probs[0]), float(probs[1])   # document, nondocument

# Request handler
class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[DocScan] {self.address_string()} - {format % args}")

    # GET: serve the HTML page
    def do_GET(self):
        if self.path in ("/", "/index.html", f"/{HTML_FILE}"):
            filepath = os.path.join(BASE_DIR, HTML_FILE)
            if not os.path.exists(filepath):
                self.send_error(404, f"{HTML_FILE} not found next to server.py")
                return
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    # POST: /classify/<version>
    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "classify":
            self.send_error(404, "Unknown endpoint")
            return

        version = parts[1]
        if version not in loaded_models:
            self._json_error(400, f"Model '{version}' is not loaded.")
            return

        content_type   = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if "multipart/form-data" not in content_type:
            self._json_error(400, "Expected multipart/form-data")
            return

        try:
            fields = parse_multipart(self.rfile, content_type, content_length)
        except Exception as e:
            self._json_error(400, f"Failed to parse upload: {e}")
            return

        if "image" not in fields:
            self._json_error(400, "No 'image' field in form data")
            return

        try:
            doc, nondoc = classify(loaded_models[version], fields["image"])
        except Exception as e:
            self._json_error(500, f"Inference error: {e}")
            return

        self._json_ok({"document": doc, "nondocument": nondoc})

    # Helpers
    def _json_ok(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, msg):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

# Main
if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    url    = f"http://localhost:{PORT}"

    print(f"\n[DocScan] Server running at {url}")
    print(f"[DocScan] Press Ctrl+C to stop\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DocScan] Stopped.")
        server.server_close()

import threading
import time
import cv2
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
import json

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.server.viewer.get_html().encode('utf-8'))
        elif self.path.startswith('/stream_'):
            cam_name = self.path.split('_')[1].split('.')[0]
            self.handle_stream(cam_name)
        elif self.path == '/api/screenshot':
            self.server.viewer.save_screenshot()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Screenshot saved')
        elif self.path == '/api/toggle_recording':
            is_recording = self.server.viewer.toggle_recording()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"recording": is_recording}).encode('utf-8'))
        else:
            self.send_error(404)

    def handle_stream(self, cam_name):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                frame = self.server.viewer.get_frame(cam_name)
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret: continue
                
                self.wfile.write(b'--frame\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.03)
        except Exception:
            pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

class WebViewer:
    def __init__(self, port=5000, output_dir="outputs/web_viewer"):
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames = {}
        self.lock = threading.Lock()
        self.recording = False
        self.writers = {}
        self.server = None
        self.thread = None
        
    def start(self):
        self.server = ThreadedHTTPServer(('0.0.0.0', self.port), StreamingHandler)
        self.server.viewer = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"Web viewer started at http://localhost:{self.port}")

    def update_frames(self, frames_dict):
        # frames_dict: {'front': img, 'left': img, 'right': img} (RGB or BGR?)
        # We assume RGB coming in, convert to BGR for OpenCV/Encoding
        with self.lock:
            for name, frame in frames_dict.items():
                # Ensure BGR for opencv
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.frames[name] = bgr_frame
                
                if self.recording:
                    if name not in self.writers:
                        self._init_writer(name, bgr_frame.shape)
                    self.writers[name].write(bgr_frame)

    def _init_writer(self, name, shape):
        h, w = shape[:2]
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        video_dir = self.output_dir / "videos" / timestamp
        video_dir.mkdir(parents=True, exist_ok=True)
        path = str(video_dir / f"{name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writers[name] = cv2.VideoWriter(path, fourcc, 30, (w, h))

    def toggle_recording(self):
        with self.lock:
            self.recording = not self.recording
            if not self.recording:
                # Stop recording
                for writer in self.writers.values():
                    writer.release()
                self.writers = {}
            return self.recording

    def save_screenshot(self):
        with self.lock:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            shot_dir = self.output_dir / "screenshots" / timestamp
            shot_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in self.frames.items():
                cv2.imwrite(str(shot_dir / f"{name}.jpg"), frame)

    def get_frame(self, name):
        with self.lock:
            return self.frames.get(name)

    def get_html(self):
        return """
        <html>
        <head>
            <title>Sim Viewer</title>
            <style>
                .container { display: flex; flex-wrap: wrap; gap: 10px; }
                .cam { border: 1px solid #ccc; }
                .controls { margin: 20px; }
                button { padding: 10px; font-size: 16px; }
            </style>
            <script>
                function screenshot() { fetch('/api/screenshot'); }
                async function toggleRec() {
                    const res = await fetch('/api/toggle_recording');
                    const data = await res.json();
                    document.getElementById('recBtn').innerText = data.recording ? "Stop Recording" : "Start Recording";
                    document.getElementById('recBtn').style.backgroundColor = data.recording ? "red" : "";
                }
            </script>
        </head>
        <body>
            <h1>Simulation Viewer</h1>
            <div class="controls">
                <button onclick="screenshot()">Screenshot</button>
                <button id="recBtn" onclick="toggleRec()">Start Recording</button>
            </div>
            <div class="container">
                <div class="cam"><h3>Front</h3><img src="/stream_front.mjpg" width="400"/></div>
                <div class="cam"><h3>Left Wrist</h3><img src="/stream_left.mjpg" width="400"/></div>
                <div class="cam"><h3>Right Wrist</h3><img src="/stream_right.mjpg" width="400"/></div>
            </div>
        </body>
        </html>
        """

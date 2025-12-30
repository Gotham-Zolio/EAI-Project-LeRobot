import os
import threading
import time
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
import json

class StreamingHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.server.viewer.get_html().encode('utf-8'))
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            status = self.server.viewer.get_status()
            self.wfile.write(json.dumps(status).encode('utf-8'))
        elif self.path.startswith('/stream_'):
            # 修复摄像头名解析，支持下划线
            cam_name = self.path[len('/stream_'):]  # 去掉前缀
            if cam_name.endswith('.mjpg'):
                cam_name = cam_name[:-5]
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
                    frame = self.server.viewer.get_placeholder(cam_name)
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                self.wfile.write(b'--frame\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.1)  # 降低帧率到10FPS，减轻卡顿
        except Exception:
            pass

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class WebViewer:
    def __init__(self, port=5000, output_dir="outputs/web_viewer"):
        self.port = port
        home_tmp = os.path.expanduser("~/tmp")
        self.output_dir = Path(home_tmp) / "outputs/web_viewer"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames = {}
        self.lock = threading.Lock()
        self.recording = False
        self.writers = {}
        self.server = None
        self.thread = None
        
        # Metadata
        self.mode = "Idle"
        self.episode = 0
        self.total_episodes = 0
        self.task = "None"
        
    def start(self):
        self.server = ThreadedHTTPServer(('0.0.0.0', self.port), StreamingHandler)
        self.server.viewer = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"Web viewer started at http://localhost:{self.port}")

    def update_status(self, mode, episode, total_episodes, task):
        with self.lock:
            self.mode = mode
            self.episode = episode
            self.total_episodes = total_episodes
            self.task = task

    def get_status(self):
        with self.lock:
            return {
                "mode": self.mode,
                "episode": self.episode,
                "total_episodes": self.total_episodes,
                "task": self.task,
                "recording": self.recording
            }

    def update_frames(self, frames_dict):
        with self.lock:
            for name, frame in frames_dict.items():
                if frame is None:
                    continue
                # 修复4通道图像（RGBA）推送问题
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]  # 丢弃alpha通道，转为RGB
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.frames[name] = bgr_frame
                if self.recording:
                    if name not in self.writers:
                        self._init_writer(name, bgr_frame.shape)
                    self.writers[name].write(bgr_frame)

    def get_placeholder(self, name):
        import numpy as np
        img = 128 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"No Signal: {name}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

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
        <!DOCTYPE html>
        <html>
        <head>
            <title>LeRobot Viewer</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #e0e0e0; margin: 0; padding: 20px; }
                h1 { color: #61dafb; margin-bottom: 10px; }
                .header { display: flex; justify-content: space-between; align-items: center; background: #2d2d2d; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
                .status-info { display: flex; gap: 20px; font-size: 1.1em; }
                .status-item { background: #3d3d3d; padding: 5px 15px; border-radius: 4px; }
                .label { color: #888; font-size: 0.9em; margin-right: 5px; }
                .value { font-weight: bold; color: #fff; }

                .container { display: flex; flex-wrap: wrap; justify-content: center; }
                .world-container { margin-bottom: 30px; }
                .small-cams { display: flex; justify-content: center; gap: 20px; }

                .cam { background: #2d2d2d; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
                .world-cam { width: 720px; }
                .cam h3 { margin: 0 0 10px 0; color: #aaa; font-size: 1em; text-align: center; }
                .cam img { border-radius: 4px; background: #000; }

                .controls { display: flex; gap: 10px; }
                button { padding: 8px 16px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; transition: background 0.2s; }
                .btn-primary { background: #007acc; color: white; }
                .btn-primary:hover { background: #005999; }
                .btn-danger { background: #d32f2f; color: white; }
                .btn-danger:hover { background: #9a0007; }
            </style>
            <script>
                function screenshot() { fetch('/api/screenshot'); }
                async function toggleRec() {
                    const res = await fetch('/api/toggle_recording');
                    const data = await res.json();
                    updateRecBtn(data.recording);
                }
                function updateRecBtn(recording) {
                    const btn = document.getElementById('recBtn');
                    btn.innerText = recording ? "Stop Recording" : "Start Recording";
                    btn.className = recording ? "btn-danger" : "btn-primary";
                }

                async function updateStatus() {
                    try {
                        const res = await fetch('/api/status');
                        const data = await res.json();
                        document.getElementById('mode').innerText = data.mode;
                        document.getElementById('task').innerText = data.task;
                        document.getElementById('episode').innerText = data.episode + " / " + data.total_episodes;
                        updateRecBtn(data.recording);
                    } catch(e) {}
                }

                setInterval(updateStatus, 1000);
            </script>
        </head>
        <body>
            <div class="header">
                <div>
                    <h1>LeRobot Viewer</h1>
                    <div class="status-info">
                        <div class="status-item"><span class="label">Mode:</span><span id="mode" class="value">Connecting...</span></div>
                        <div class="status-item"><span class="label">Task:</span><span id="task" class="value">-</span></div>
                        <div class="status-item"><span class="label">Episode:</span><span id="episode" class="value">-</span></div>
                    </div>
                </div>
                <div class="controls">
                    <button class="btn-primary" onclick="screenshot()">Screenshot</button>
                    <button id="recBtn" class="btn-primary" onclick="toggleRec()">Start Recording</button>
                </div>
            </div>

            <!-- World Camera -->
            <div class="container world-container">
                <div class="cam world-cam">
                    <h3>World Camera</h3>
                    <img src="/stream_world.mjpg" width="720" height="540"/>
                </div>
            </div>

            <!-- Small Cameras -->
            <div class="container small-cams">
                <div class="cam"><h3>Front Camera</h3><img src="/stream_front.mjpg" width="420" height="315"/></div>
                <div class="cam"><h3>Left Wrist</h3><img src="/stream_left_wrist.mjpg" width="420" height="315"/></div>
                <div class="cam"><h3>Right Wrist</h3><img src="/stream_right_wrist.mjpg" width="420" height="315"/></div>
            </div>
        </body>
        </html>
        """

#code by conduttanza
#
#created the 17/12/2025

import cv2, pygame, time, numpy as np
from threading import Thread, Lock
from functions_logic import Config
import urllib.request
import urllib.parse
import socket

config = Config()

class Image:

    def __init__(self):
        self.stream_url = config.stream_url
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = Lock()

        if self.stream_url:
            Thread(target=self._mjpeg_thread, daemon=True).start()
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.side_x)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.side_y)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot use camera")
            Thread(target=self.update, daemon=True).start()

    def _mjpeg_thread(self):
        total_bytes = b""
        try:
            # ensure URL has a scheme
            stream_url = config.stream_url
            parsed = urllib.parse.urlparse(stream_url)
            if not parsed.scheme:
                stream_url = stream_url
                parsed = urllib.parse.urlparse(stream_url)
                print('Note: stream URL had no scheme, trying with http://')

            # quick DNS / address check to provide clearer error messages
            try:
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                socket.getaddrinfo(host, port)
            except Exception as e:
                print(f"Cannot resolve host {parsed.hostname}:{port} ->", e)
                self.running = False
                return

            req = urllib.request.Request(stream_url, headers={"User-Agent": "Mozilla/5.0"})
            stream = urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print("Cannot open stream URL:", e)
            self.running = False
            return
        while self.running:
            try:
                total_bytes += stream.read(1024)
                a = total_bytes.find(b'\xff\xd8')
                b = total_bytes.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = total_bytes[a:b+2]
                    total_bytes = total_bytes[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.ret = True
                            self.frame = frame
            except Exception:
                continue

    def update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = True
                        self.frame = frame
            except cv2.error:
                time.sleep(config.delay)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.ret else None

    def get_cap(self):
        return getattr(self, 'cap', None)

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

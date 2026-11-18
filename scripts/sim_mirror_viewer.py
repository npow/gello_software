"""Kinematic sim preview of GELLO -> YAM mapping, viewable in a browser.

Reads the GELLO through the same agent config used for hardware, applies the
commanded joints directly to the MuJoCo YAM model (no physics, no robot),
and serves a live MJPEG stream plus joint readout over HTTP.

Usage:
    MUJOCO_GL=egl python scripts/sim_mirror_viewer.py \
        --config configs/yam_auto_generated.yaml --port 8000
"""

import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from gello.utils.launch_utils import instantiate_from_dict

XML = "third_party/mujoco_menagerie/i2rt_yam/yam.xml"

state = {"jpeg": b"", "cmd": np.zeros(7)}
lock = threading.Lock()


def render_loop(agent, hz: float) -> None:
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 640)
    finger_ids = [m.joint(n).qposadr[0] for n in ("left_finger", "right_finger")]
    finger_rng = [m.jnt_range[m.joint(n).id] for n in ("left_finger", "right_finger")]
    cams = []
    for az in (135.0, 45.0):
        c = mujoco.MjvCamera()
        c.lookat[:] = [0.15, 0.0, 0.25]
        c.distance, c.azimuth, c.elevation = 1.1, az, -25.0
        cams.append(c)
    obs = {"joint_positions": np.zeros(7)}
    while True:
        cmd = np.asarray(agent.act(obs), dtype=float)
        d.qpos[:6] = cmd[:6]
        g = float(np.clip(cmd[6], 0.0, 1.0))
        for adr, rng in zip(finger_ids, finger_rng):
            d.qpos[adr] = rng[0] + g * (rng[1] - rng[0])
        mujoco.mj_forward(m, d)
        frames = []
        for c in cams:
            r.update_scene(d, camera=c)
            frames.append(r.render())
        buf = io.BytesIO()
        Image.fromarray(np.concatenate(frames, axis=1)).save(buf, "JPEG", quality=80)
        with lock:
            state["jpeg"] = buf.getvalue()
            state["cmd"] = cmd
        time.sleep(1.0 / hz)


PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>YAM Sim Mirror</title><style>body{background:#111;color:#eee;font:14px monospace;margin:16px}
img{max-width:100%}</style></head><body>
<img src="/stream"><pre id=j></pre>
<script>setInterval(async()=>{document.getElementById('j').textContent=
await (await fetch('/joints')).text()},200)</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/joints":
            with lock:
                cmd = state["cmd"]
            names = ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
            body = "\n".join(f"{n}: {v:+.3f}" for n, v in zip(names, cmd)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = state["jpeg"]
                    self.wfile.write(
                        b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yam_auto_generated.yaml")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--hz", type=float, default=30.0)
    args = ap.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    agent = instantiate_from_dict(cfg["agent"])
    threading.Thread(target=render_loop, args=(agent, args.hz), daemon=True).start()
    print(f"Viewer: http://0.0.0.0:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

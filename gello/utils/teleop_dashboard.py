"""Rerun Operator Dashboard & Episodic Data Recorder for YAM / GELLO Teleop."""

from __future__ import annotations

import http.server
import json
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure venv bin is in PATH so rerun can locate its binary
_venv_bin = str(Path(sys.executable).parent)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_venv_bin}:{os.environ.get('PATH', '')}"

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from gello.cameras.v4l2_camera import V4L2Camera

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_network_addrs(web_port: int, grpc_port: int) -> Dict[str, str]:
    """Discover Tailscale DNS, MagicDNS, Tailnet IP, and local addresses."""
    addrs = {
        "localhost": f"http://127.0.0.1:{web_port}",
    }
    try:
        res = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            data = json.loads(res.stdout)
            self_node = data.get("Self", {})
            dns_name = self_node.get("DNSName", "").rstrip(".")
            host_name = self_node.get("HostName", "")
            ips = self_node.get("TailscaleIPs", [])

            if dns_name:
                addrs["tailscale_dns"] = f"http://{dns_name}:{web_port}"
                addrs["tailscale_dns_grpc"] = (
                    f"rerun+http://{dns_name}:{grpc_port}/proxy"
                )
                addrs["dns_host"] = dns_name
            if host_name:
                addrs["tailscale_name"] = f"http://{host_name}:{web_port}"
            if ips:
                addrs["tailnet_ip"] = f"http://{ips[0]}:{web_port}"
                addrs["tailnet_ip_grpc"] = f"rerun+http://{ips[0]}:{grpc_port}/proxy"
    except Exception:
        pass
    return addrs


class _AutoRedirectProxy(http.server.SimpleHTTPRequestHandler):
    """HTTP proxy that forwards requests to Rerun's web viewer and auto-injects ?url= on root requests."""

    grpc_port: int = 9876
    internal_port: int = 9091
    tailscale_dns: Optional[str] = None

    def log_message(self, format, *args):
        # Suppress noisy HTTP asset logs from terminal
        pass

    def do_HEAD(self):
        self._handle_proxy(is_head=True)

    def do_GET(self):
        self._handle_proxy(is_head=False)

    def _handle_proxy(self, is_head: bool = False):
        host_header = self.headers.get("Host", f"127.0.0.1:{self.server.server_port}")
        if host_header.startswith("["):
            host = (
                host_header[: host_header.find("]") + 1]
                if "]" in host_header
                else host_header
            )
        else:
            host = host_header.split(":")[0]

        parsed = urllib.parse.urlparse(self.path)
        # If user opened root path without ?url=, redirect with ?url= matching the host accessed
        if parsed.path == "/" and not parsed.query:
            target_host = host if host else (self.tailscale_dns or "127.0.0.1")
            grpc_url = f"rerun+http://{target_host}:{self.grpc_port}/proxy"
            encoded_url = urllib.parse.quote(grpc_url, safe="")
            target = f"/?url={encoded_url}"
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return

        # If URL contains unencoded '+', redirect to percent-encoded '%2B' so JS URLSearchParams doesn't turn it into a space
        if "+" in self.path:
            clean_path = self.path.replace("+", "%2B")
            self.send_response(302)
            self.send_header("Location", clean_path)
            self.end_headers()
            return

        # Proxy static asset requests (.wasm, .js, .html) to internal Rerun web viewer
        url = f"http://127.0.0.1:{self.internal_port}{self.path}"
        req = urllib.request.Request(url, headers=dict(self.headers))
        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                content = resp.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                if not is_head:
                    self.wfile.write(content)
        except Exception as e:
            self.send_error(502, f"Proxy error: {e}")


class TerminalKeyboard:
    """Non-blocking terminal keyboard reader."""

    def __init__(self):
        self._is_tty = sys.stdin.isatty()
        self._old_settings = None
        if self._is_tty:
            try:
                self._old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self._is_tty = False

    def get_key(self) -> Optional[str]:
        if not self._is_tty:
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            return sys.stdin.read(1)
        return None

    def close(self):
        if self._is_tty and self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass


class RerunDashboard:
    """Live Operator Dashboard via Rerun + Synchronized Episodic Data Recorder."""

    def __init__(
        self,
        output_dir: str = "data/episodes",
        use_web: bool = False,
        web_port: int = 9090,
        grpc_port: int = 9876,
        bind_host: str = "0.0.0.0",
        enable_cameras: bool = True,
        save_rrd: bool = True,
        camera_fps: int = 15,
        bimanual: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bimanual = bimanual
        self.enable_cameras = enable_cameras
        self.save_rrd = save_rrd
        self.camera_fps = camera_fps
        self.bind_host = bind_host
        self.web_port = web_port
        self.grpc_port = grpc_port
        self._proxy_server: Optional[http.server.ThreadingHTTPServer] = None

        # Initialize terminal keyboard
        self.kb = TerminalKeyboard()

        # Cameras dict
        self.cameras: Dict[str, V4L2Camera] = {}
        if self.enable_cameras:
            # Map cameras to working video endpoints
            cam_map = {
                "left": (
                    "/dev/video8"
                    if Path("/dev/video8").exists()
                    else "/dev/yam-cameras/left"
                ),
                "middle": (
                    "/dev/video16"
                    if Path("/dev/video16").exists()
                    else "/dev/yam-cameras/middle"
                ),
                "right": (
                    "/dev/video2"
                    if Path("/dev/video2").exists()
                    else "/dev/yam-cameras/right"
                ),
            }
            for name, dev in cam_map.items():
                if Path(dev).exists():
                    try:
                        self.cameras[name] = V4L2Camera(
                            dev, width=640, height=480, fps=camera_fps
                        )
                        print(f"  {GREEN}✓{RESET} Connected camera '{name}' ({dev})")
                    except Exception as e:
                        print(
                            f"  {YELLOW}⚠{RESET} Could not open camera '{name}' ({dev}): {e}"
                        )

        # Initialize Rerun
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )

        rr.init("yam_teleop")

        # Always serve gRPC with wildcard CORS so remote/tailnet viewers can connect
        try:
            rr.serve_grpc(grpc_port=grpc_port, cors_allow_origin=["*"])
        except Exception as e:
            print(f"  {YELLOW}⚠{RESET} Rerun gRPC notice: {e}")

        # If web requested or headless, start web viewer and proxy
        if use_web or not has_display:
            internal_port = web_port + 1
            try:
                rr.serve_web_viewer(web_port=internal_port, open_browser=False)

                net_info = get_network_addrs(web_port, grpc_port)

                class ConfiguredProxy(_AutoRedirectProxy):
                    pass

                ConfiguredProxy.grpc_port = grpc_port
                ConfiguredProxy.internal_port = internal_port
                ConfiguredProxy.tailscale_dns = net_info.get("dns_host")

                self._proxy_server = http.server.ThreadingHTTPServer(
                    (bind_host, web_port), ConfiguredProxy
                )
                t = threading.Thread(
                    target=self._proxy_server.serve_forever, daemon=True
                )
                t.start()

                print(
                    f"\n{BOLD}{CYAN}🌐 Rerun Web Dashboard Active (Tailscale DNS):{RESET}"
                )
                if "tailscale_dns" in net_info:
                    print(
                        f"  • {BOLD}Tailscale DNS:{RESET} {net_info['tailscale_dns']}"
                    )
                if "tailscale_name" in net_info:
                    print(
                        f"  • {BOLD}MagicDNS:{RESET}      {net_info['tailscale_name']}"
                    )
                if "tailnet_ip" in net_info:
                    print(f"  • {BOLD}Tailnet IP:{RESET}    {net_info['tailnet_ip']}")
                print(f"  • {BOLD}Localhost:{RESET}     {net_info['localhost']}")
                if "tailscale_dns_grpc" in net_info:
                    print(
                        f"  • {BOLD}Native App:{RESET}    rerun {net_info['tailscale_dns_grpc']}\n"
                    )
                elif "tailnet_ip_grpc" in net_info:
                    print(
                        f"  • {BOLD}Native App:{RESET}    rerun {net_info['tailnet_ip_grpc']}\n"
                    )
                else:
                    print(
                        f"  • {BOLD}Native App:{RESET}    rerun rerun+http://127.0.0.1:{grpc_port}/proxy\n"
                    )
            except Exception as e:
                print(f"  {YELLOW}⚠{RESET} Web viewer notice: {e}")
        else:
            try:
                rr.spawn()
                print(
                    f"{BOLD}{CYAN}🖥️ Rerun native viewer spawned on local display.{RESET}"
                )
            except Exception as e:
                print(
                    f"  {YELLOW}⚠{RESET} Native spawn failed ({e}), falling back to web viewer..."
                )
                internal_port = web_port + 1
                rr.serve_web_viewer(web_port=internal_port, open_browser=False)

                net_info = get_network_addrs(web_port, grpc_port)

                class ConfiguredProxy(_AutoRedirectProxy):
                    pass

                ConfiguredProxy.grpc_port = grpc_port
                ConfiguredProxy.internal_port = internal_port
                ConfiguredProxy.tailscale_dns = net_info.get("dns_host")

                self._proxy_server = http.server.ThreadingHTTPServer(
                    (bind_host, web_port), ConfiguredProxy
                )
                t = threading.Thread(
                    target=self._proxy_server.serve_forever, daemon=True
                )
                t.start()

        # Setup Blueprint
        views = []
        if self.cameras:
            cam_views = [
                rrb.Spatial2DView(
                    origin=f"/cameras/{name}", name=f"{name.capitalize()} Camera"
                )
                for name in self.cameras.keys()
            ]
            views.append(rrb.Horizontal(*cam_views, column_shares=[1] * len(cam_views)))

        time_views = [
            rrb.TimeSeriesView(origin="/teleop/left_arm", name="Left Arm Joints"),
        ]
        if self.bimanual:
            time_views.append(
                rrb.TimeSeriesView(origin="/teleop/right_arm", name="Right Arm Joints")
            )
        time_views.append(
            rrb.TimeSeriesView(origin="/teleop/telemetry", name="Telemetry & Status")
        )

        views.append(rrb.Horizontal(*time_views, column_shares=[1] * len(time_views)))

        blueprint = rrb.Blueprint(
            rrb.Vertical(*views),
            collapse_panels=True,
        )
        self.blueprint = blueprint
        rr.send_blueprint(blueprint)

        # Recording state
        self.is_recording = False
        self.episode_idx = self._find_next_episode_idx()
        self.current_episode_dir: Optional[Path] = None
        self.video_writers: Dict[str, cv2.VideoWriter] = {}
        self.episode_data: Dict[str, List] = {}
        self._episode_rrd_stream: Optional[rr.RecordingStream] = None

        # Timing
        self.step_idx = 0
        self.episode_step_idx = 0
        self.last_step_time = time.time()
        self.hz = 30.0

        print(
            f"\n{BOLD}{CYAN}=== Operator Dashboard & Teleop Recorder Ready ==={RESET}"
        )
        print("  Controls:")
        print(
            f"    {BOLD}[Space]{RESET} or {BOLD}[s]{RESET} : Start / Stop recording episode"
        )
        print(f"    {BOLD}[d]{RESET}         : Discard current episode")
        print(f"    {BOLD}[q]{RESET}         : Quit teleoperation\n")

    def _find_next_episode_idx(self) -> int:
        existing = [d.name for d in self.output_dir.glob("episode_*") if d.is_dir()]
        indices = []
        for name in existing:
            try:
                indices.append(int(name.split("_")[-1]))
            except ValueError:
                pass
        return max(indices, default=0) + 1

    def _start_recording(self):
        self.is_recording = True
        self.episode_step_idx = 0
        self.current_episode_dir = self.output_dir / f"episode_{self.episode_idx:04d}"
        self.current_episode_dir.mkdir(parents=True, exist_ok=True)

        # Stream directly to Rerun .rrd file if enabled
        if self.save_rrd:
            try:
                rrd_path = str(self.current_episode_dir / "recording.rrd")
                self._episode_rrd_stream = rr.RecordingStream(
                    "yam_teleop", recording_id=f"episode_{self.episode_idx:04d}"
                )
                self._episode_rrd_stream.save(rrd_path)
                self._episode_rrd_stream.send_blueprint(self.blueprint)
            except Exception as e:
                print(f"  {YELLOW}⚠ Failed to init .rrd stream: {e}{RESET}")
                self._episode_rrd_stream = None

        # Initialize video writers for any active cameras
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writers = {}
        for name in self.cameras.keys():
            vid_path = str(self.current_episode_dir / f"{name}.mp4")
            self.video_writers[name] = cv2.VideoWriter(
                vid_path, fourcc, 30.0, (640, 480)
            )

        # Reset episode buffer
        self.episode_data = {
            "step": [],
            "timestamp": [],
            "follower_joints": [],
            "leader_actions": [],
            "tracking_error": [],
        }
        print(
            f"\n{BOLD}{GREEN}● STARTED RECORDING: Episode {self.episode_idx:04d}{RESET}"
        )

    def _stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        # Finalize video writers
        for out in self.video_writers.values():
            out.release()
        self.video_writers.clear()

        # Finalize Rerun .rrd file
        if self._episode_rrd_stream is not None:
            try:
                self._episode_rrd_stream.disconnect()
            except Exception:
                pass
            self._episode_rrd_stream = None

        # Save trajectory data
        if self.current_episode_dir is not None:
            traj_path = self.current_episode_dir / "trajectory.npz"
            np.savez_compressed(
                traj_path,
                step=np.array(self.episode_data["step"]),
                timestamp=np.array(self.episode_data["timestamp"]),
                follower_joints=np.array(self.episode_data["follower_joints"]),
                leader_actions=np.array(self.episode_data["leader_actions"]),
                tracking_error=np.array(self.episode_data["tracking_error"]),
            )

            # Save metadata
            meta_path = self.current_episode_dir / "metadata.json"
            meta = {
                "episode_id": self.episode_idx,
                "num_frames": self.episode_step_idx,
                "bimanual": self.bimanual,
                "cameras": list(self.cameras.keys()),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(
                f"\n{BOLD}{GREEN}✓ SAVED: Episode {self.episode_idx:04d} ({self.episode_step_idx} frames) -> {self.current_episode_dir}{RESET}"
            )
            self.episode_idx += 1
            self.current_episode_dir = None

    def _discard_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        for out in self.video_writers.values():
            out.release()
        self.video_writers.clear()

        if self._episode_rrd_stream is not None:
            try:
                self._episode_rrd_stream.disconnect()
            except Exception:
                pass
            self._episode_rrd_stream = None

        if self.current_episode_dir is not None and self.current_episode_dir.exists():
            import shutil

            shutil.rmtree(self.current_episode_dir, ignore_errors=True)
            print(f"\n{BOLD}{RED}✗ DISCARDED: Episode {self.episode_idx:04d}{RESET}")
        self.current_episode_dir = None

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Update dashboard and logging on each step. Returns 'quit' if requested."""
        now = time.time()
        dt = now - self.last_step_time
        if dt > 0:
            self.hz = 0.9 * self.hz + 0.1 * (1.0 / dt)
        self.last_step_time = now

        self.step_idx += 1
        rr.set_time("step", sequence=self.step_idx)
        rr.set_time("time", timestamp=now)

        # Check key inputs
        key = self.kb.get_key()
        if key in (" ", "s", "S"):
            if self.is_recording:
                self._stop_recording()
            else:
                self._start_recording()
        elif key in ("d", "D"):
            self._discard_recording()
        elif key in ("q", "Q"):
            if self.is_recording:
                self._stop_recording()
            print("\nQuit requested.")
            return "quit"

        # Follower & Leader state
        follower_joints = np.array(obs.get("joint_positions", []))
        leader_actions = np.array(action)
        tracking_error = (
            np.abs(follower_joints - leader_actions)
            if len(follower_joints) == len(leader_actions)
            else np.zeros_like(leader_actions)
        )

        # Log to Rerun
        rr.log("teleop/telemetry/hz", rr.Scalars(self.hz))
        rr.log(
            "teleop/telemetry/is_recording",
            rr.Scalars(1.0 if self.is_recording else 0.0),
        )
        rr.log(
            "teleop/telemetry/max_tracking_error_deg",
            rr.Scalars(
                float(np.degrees(np.max(tracking_error)))
                if len(tracking_error)
                else 0.0
            ),
        )

        # Log joint series
        num_left = 7 if self.bimanual else len(follower_joints)
        for i in range(min(num_left, len(follower_joints))):
            rr.log(f"teleop/left_arm/j{i+1}_actual", rr.Scalars(follower_joints[i]))
            if i < len(leader_actions):
                rr.log(f"teleop/left_arm/j{i+1}_target", rr.Scalars(leader_actions[i]))

        if self.bimanual and len(follower_joints) >= 14:
            for i in range(7, 14):
                rr.log(
                    f"teleop/right_arm/j{i-6}_actual", rr.Scalars(follower_joints[i])
                )
                if i < len(leader_actions):
                    rr.log(
                        f"teleop/right_arm/j{i-6}_target", rr.Scalars(leader_actions[i])
                    )

        # Read and log cameras
        frames: Dict[str, np.ndarray] = {}
        for name, cam in self.cameras.items():
            try:
                rgb, _ = cam.read()
                frames[name] = rgb
                rr.log(f"cameras/{name}", rr.Image(rgb))
            except Exception:
                pass

        # Handle active recording
        if self.is_recording:
            self.episode_step_idx += 1
            self.episode_data["step"].append(self.episode_step_idx)
            self.episode_data["timestamp"].append(now)
            self.episode_data["follower_joints"].append(follower_joints)
            self.episode_data["leader_actions"].append(leader_actions)
            self.episode_data["tracking_error"].append(tracking_error)

            # Stream to episode .rrd file
            if self._episode_rrd_stream is not None:
                self._episode_rrd_stream.set_time(
                    "step", sequence=self.episode_step_idx
                )
                self._episode_rrd_stream.set_time("time", timestamp=now)
                self._episode_rrd_stream.log("teleop/telemetry/hz", rr.Scalars(self.hz))
                self._episode_rrd_stream.log(
                    "teleop/telemetry/max_tracking_error_deg",
                    rr.Scalars(
                        float(np.degrees(np.max(tracking_error)))
                        if len(tracking_error)
                        else 0.0
                    ),
                )
                for j in range(min(num_left, len(follower_joints))):
                    self._episode_rrd_stream.log(
                        f"teleop/left_arm/j{j+1}_actual", rr.Scalars(follower_joints[j])
                    )
                    if j < len(leader_actions):
                        self._episode_rrd_stream.log(
                            f"teleop/left_arm/j{j+1}_target",
                            rr.Scalars(leader_actions[j]),
                        )
                if self.bimanual and len(follower_joints) >= 14:
                    for j in range(7, 14):
                        self._episode_rrd_stream.log(
                            f"teleop/right_arm/j{j-6}_actual",
                            rr.Scalars(follower_joints[j]),
                        )
                        if j < len(leader_actions):
                            self._episode_rrd_stream.log(
                                f"teleop/right_arm/j{j-6}_target",
                                rr.Scalars(leader_actions[j]),
                            )
                for name, rgb in frames.items():
                    _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self._episode_rrd_stream.log(
                        f"cameras/{name}",
                        rr.EncodedImage(
                            contents=buf.tobytes(), media_type="image/jpeg"
                        ),
                    )

            # Write to videos (convert RGB to BGR for cv2 VideoWriter)
            for name, rgb in frames.items():
                if name in self.video_writers:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    self.video_writers[name].write(bgr)

            status_line = (
                f"\r{BOLD}{GREEN}[● RECORDING]{RESET} Episode {self.episode_idx:04d} | "
                f"Frames: {self.episode_step_idx:4d} ({(self.episode_step_idx/30.0):.1f}s) | "
                f"{self.hz:.1f} Hz | Max Err: {np.degrees(np.max(tracking_error)):.1f}° | "
                f"Keys: [Space]=Stop, [d]=Discard"
            )
        else:
            status_line = (
                f"\r{BOLD}{CYAN}[○ IDLE]{RESET} Next: Episode {self.episode_idx:04d} | "
                f"{self.hz:.1f} Hz | "
                f"Keys: [Space]=Record, [q]=Quit"
            )

        print(status_line, end="", flush=True)
        return None

    def close(self):
        if self.is_recording:
            self._stop_recording()
        if self._proxy_server is not None:
            try:
                self._proxy_server.shutdown()
                self._proxy_server.server_close()
            except Exception:
                pass
            self._proxy_server = None
        self.kb.close()
        for cam in self.cameras.values():
            cam.close()
        print("\nDashboard closed.")

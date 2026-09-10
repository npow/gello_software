import atexit
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    use_save_interface: bool = False
    """Enable legacy saving data with keyboard interface."""

    use_rerun: bool = False
    """Enable Rerun live operator dashboard and episodic recorder."""

    rerun_web: bool = False
    """Open Rerun dashboard in browser (web viewer) instead of native desktop window."""

    enable_cameras: bool = True
    """Enable streaming and recording from /dev/yam-cameras/* in Rerun."""

    data_dir: str = "data/episodes"
    """Output directory for recorded episodes."""

    bind_host: str = "0.0.0.0"
    """Host IP to bind web viewer, gRPC stream, and robot servers to (default: 0.0.0.0 for Tailnet access)."""

    web_port: int = 9090
    """Port for Rerun web operator dashboard."""

    grpc_port: int = 9876
    """Port for Rerun gRPC telemetry stream."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def preflight_can_check(channel: str):
    """Ensure CAN interface is UP and clear any latched faults before robot init."""
    import subprocess

    try:
        res = subprocess.run(
            ["ip", "link", "show", channel],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and (
            "state DOWN" in res.stdout
            or ("<NOARP" in res.stdout and "UP" not in res.stdout)
        ):
            print(
                f"[Auto-Recovery] CAN interface '{channel}' is DOWN. Bringing it UP..."
            )
            subprocess.run(
                [
                    "sudo",
                    "ip",
                    "link",
                    "set",
                    channel,
                    "up",
                    "type",
                    "can",
                    "bitrate",
                    "1000000",
                ],
                check=False,
            )
    except Exception:
        pass

    try:
        import can

        bus = can.interface.Bus(channel=channel, interface="socketcan")
        print(f"[Auto-Recovery] Pre-clearing latched motor errors on '{channel}'...")
        for mid in range(1, 8):
            bus.send(
                can.Message(
                    arbitration_id=mid, data=[0xFF] * 7 + [0xFB], is_extended_id=False
                )
            )
        time.sleep(0.02)
        while bus.recv(timeout=0.005) is not None:
            pass
        bus.shutdown()
    except Exception:
        pass


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    if bimanual:
        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

    # Pre-flight CAN check and fault clearing
    can_channels = []
    if "channel" in left_robot_cfg:
        can_channels.append(left_robot_cfg["channel"])
    if bimanual and "channel" in right_robot_cfg:
        can_channels.append(right_robot_cfg["channel"])
    for ch in can_channels:
        preflight_can_check(ch)

    try:
        left_robot = instantiate_from_dict(left_robot_cfg)

        if bimanual:
            from gello.robots.robot import BimanualRobot

            right_robot = instantiate_from_dict(right_robot_cfg)
            robot = BimanualRobot(left_robot, right_robot)

            # For bimanual, use the left config for general settings (hz, etc.)
            cfg = left_cfg
        else:
            robot = left_robot
            cfg = left_cfg
    except Exception as e:
        print(f"\n[Launch Error] Robot initialization failed: {e}")
        cleanup()
        import os

        os._exit(1)

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = args.bind_host

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready (connect via localhost loopback)
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, "127.0.0.1")
        print("Hardware server ready!")

        # Create client to communicate with hardware locally
        robot_client = ZMQClientRobot(port=hardware_port, host="127.0.0.1")

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface / rerun dashboard
    save_interface = None
    dashboard = None

    if args.use_rerun:
        from gello.utils.teleop_dashboard import RerunDashboard

        dashboard = RerunDashboard(
            output_dir=args.data_dir,
            use_web=args.rerun_web,
            web_port=args.web_port,
            grpc_port=args.grpc_port,
            bind_host=args.bind_host,
            enable_cameras=args.enable_cameras,
            bimanual=bimanual,
        )
        save_interface = dashboard
    elif args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=Path(args.left_config_path).parents[1] / "data",
            agent_name=agent.__class__.__name__,
            expand_user=True,
        )

    try:
        # Run main control loop (suppress print_timing if RerunDashboard is active)
        run_control_loop(env, agent, save_interface, print_timing=(dashboard is None))
    finally:
        if dashboard is not None:
            dashboard.close()


if __name__ == "__main__":
    main()

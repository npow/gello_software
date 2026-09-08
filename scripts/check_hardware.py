#!/usr/bin/env python3
"""Hardware diagnostic and pre-flight check script for YAM GELLO setups.

Checks:
1. CAN network interfaces (can_left, can_right)
2. Follower DM motors (IDs 1-7 on each arm), with automatic fault clearing
3. Leader Dynamixel serial ports and motors (IDs 1-7)
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import can
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from serial.serialutil import SerialException

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

MOTOR_ERROR_CODES = {
    0x0: "disabled",
    0x1: "normal",
    0x8: "over voltage",
    0x9: "under voltage",
    0xA: "over current",
    0xB: "mosfet over temp",
    0xC: "motor over temp",
    0xD: "loss communication",
    0xE: "overload",
}


def check_can_interface(iface: str) -> bool:
    """Check if CAN interface exists and is UP."""
    try:
        res = subprocess.run(
            ["ip", "link", "show", iface],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            print(f"  {RED}✗ Interface {iface} not found in system!{RESET}")
            return False
        if "state UP" in res.stdout or "<NOARP,UP" in res.stdout:
            print(f"  {GREEN}✓ {iface} is UP{RESET}")
            return True
        else:
            print(
                f"  {RED}✗ {iface} is DOWN!{RESET} Run: sudo bash scripts/reset_all_can.sh"
            )
            return False
    except Exception as e:
        print(f"  {RED}✗ Error checking {iface}: {e}{RESET}")
        return False


def test_and_recover_can_motors(channel: str, clear_faults: bool = True):
    """Query all 7 DM motors on a CAN channel and clear faults if needed."""
    try:
        bus = can.interface.Bus(channel=channel, interface="socketcan")
    except Exception as e:
        print(f"  {RED}✗ Cannot open {channel}: {e}{RESET}")
        return False

    all_ok = True
    print(f"  Scanning motors 1-7 on {channel}:")

    for motor_id in range(1, 8):
        msg = can.Message(
            arbitration_id=motor_id,
            data=[0xFF] * 7 + [0xFC],
            is_extended_id=False,
        )
        responded = False
        err_msg = "No response"
        err_code = -1

        for attempt in range(3):
            try:
                bus.send(msg)
                start = time.time()
                while time.time() - start < 0.04:
                    r = bus.recv(timeout=0.01)
                    if r and r.arbitration_id == motor_id + 16:
                        responded = True
                        if len(r.data) >= 1:
                            err_code = (r.data[0] >> 4) & 0x0F
                            err_msg = MOTOR_ERROR_CODES.get(
                                err_code, f"code 0x{err_code:X}"
                            )
                        break
                if responded:
                    break
            except Exception:
                pass
            time.sleep(0.005)

        if responded:
            if err_code == 0x1 or err_code == 0x0:
                print(f"    Joint {motor_id}: {GREEN}OK{RESET} ({err_msg})")
            else:
                print(
                    f"    Joint {motor_id}: {YELLOW}FAULT: {err_msg} (0x{err_code:X}){RESET}",
                    end="",
                )
                if clear_faults:
                    clear_msg = can.Message(
                        arbitration_id=motor_id,
                        data=[0xFF] * 7 + [0xFB],
                        is_extended_id=False,
                    )
                    try:
                        bus.send(clear_msg)
                        time.sleep(0.01)
                        bus.send(msg)
                        time.sleep(0.01)
                        r = bus.recv(timeout=0.02)
                        new_code = (r.data[0] >> 4) & 0x0F if r else -1
                        if new_code == 0x1:
                            print(f" -> {GREEN}CLEARED (OK){RESET}")
                        else:
                            print(f" -> {YELLOW}Reset sent{RESET}")
                    except Exception as ce:
                        print(f" -> Clear failed: {ce}")
                else:
                    print()
        else:
            print(f"    Joint {motor_id}: {RED}NO RESPONSE{RESET}")
            all_ok = False

    bus.shutdown()
    return all_ok


def check_dxl_leader(port: str, expected_motors: int = 7):
    """Check Dynamixel leader motors on serial port."""
    if not Path(port).exists():
        print(f"  {RED}✗ Serial port does not exist: {port}{RESET}")
        return False

    ph = PortHandler(port)
    try:
        if not ph.openPort():
            print(f"  {RED}✗ Cannot open serial port: {port}{RESET}")
            return False
    except Exception as e:
        print(f"  {RED}✗ Cannot open {port}: {e}{RESET}")
        return False

    try:
        ph.setBaudRate(57600)
        pkh = PacketHandler(2.0)

        print(f"  Pinging Dynamixel motors 1-{expected_motors} on {port}:")
        all_ok = True
        for mid in range(1, expected_motors + 1):
            try:
                model, res, err = pkh.ping(ph, mid)
                if res == COMM_SUCCESS:
                    print(f"    Servo {mid}: {GREEN}OK{RESET} (Model: {model})")
                else:
                    print(f"    Servo {mid}: {RED}NO RESPONSE{RESET} (Err: {err})")
                    all_ok = False
            except SerialException as se:
                print(
                    f"    {YELLOW}Port busy or read error on Servo {mid} (is launch_yaml currently running?): {se}{RESET}"
                )
                return True
    finally:
        ph.closePort()

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose YAM physical arms and GELLO leaders"
    )
    parser.add_argument(
        "--left-can", default="can_left", help="Left follower CAN interface"
    )
    parser.add_argument(
        "--right-can", default="can_right", help="Right follower CAN interface"
    )
    parser.add_argument(
        "--left-serial",
        default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEQBRC-if00-port0",
        help="Left leader serial port",
    )
    parser.add_argument(
        "--right-serial",
        default="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEQC10-if00-port0",
        help="Right leader serial port",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}=== YAM & GELLO PRE-FLIGHT HARDWARE DIAGNOSTICS ==={RESET}\n")

    # 1. CAN Network Interfaces
    print(f"{BOLD}[1/4] CAN Network Interfaces{RESET}")
    left_can_up = check_can_interface(args.left_can)
    right_can_up = check_can_interface(args.right_can)

    # 2. Follower Motors
    print(f"\n{BOLD}[2/4] Left Follower Arm ({args.left_can}){RESET}")
    left_motors_ok = (
        test_and_recover_can_motors(args.left_can) if left_can_up else False
    )

    print(f"\n{BOLD}[3/4] Right Follower Arm ({args.right_can}){RESET}")
    right_motors_ok = (
        test_and_recover_can_motors(args.right_can) if right_can_up else False
    )

    # 3. Leaders
    print(f"\n{BOLD}[4/4] Leader Arm Servos{RESET}")
    print("Left Leader:")
    left_leader_ok = check_dxl_leader(args.left_serial)
    print("Right Leader:")
    right_leader_ok = check_dxl_leader(args.right_serial)

    # Summary
    print(f"\n{BOLD}{CYAN}=== DIAGNOSTICS SUMMARY ==={RESET}")
    status = lambda ok: f"{GREEN}READY{RESET}" if ok else f"{RED}ACTION REQUIRED{RESET}"
    print(f"  Left Follower Arm:  {status(left_can_up and left_motors_ok)}")
    print(f"  Right Follower Arm: {status(right_can_up and right_motors_ok)}")
    print(f"  Left GELLO Leader:  {status(left_leader_ok)}")
    print(f"  Right GELLO Leader: {status(right_leader_ok)}")

    if (
        left_can_up
        and right_can_up
        and left_motors_ok
        and right_motors_ok
        and left_leader_ok
        and right_leader_ok
    ):
        print(f"\n{BOLD}{GREEN}✓ All systems ready for launch!{RESET}")
        print("Run command:")
        print(
            "  python experiments/launch_yaml.py --left-config-path configs/yam_auto_generated.yaml --right-config-path configs/yam_auto_generated_right.yaml\n"
        )
    else:
        print(f"\n{BOLD}{YELLOW}⚠ Please fix the errors above before launching.{RESET}")
        if not (left_can_up and right_can_up):
            print("  - To bring up CAN interfaces: sudo bash scripts/reset_all_can.sh")
        if not (left_motors_ok and right_motors_ok):
            print(
                "  - If follower motors did not respond: check 24V/48V power switch to the arms."
            )
        if not (left_leader_ok and right_leader_ok):
            print(
                "  - If leader servos did not respond: check USB connection or 12V U2D2 power."
            )
        print()


if __name__ == "__main__":
    main()

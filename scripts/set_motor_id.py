"""Script to change Dynamixel motor ID and ensure 57600 baudrate for GELLO setups.

Usage:
    Connect ONLY ONE motor at a time to the U2D2 / USB converter.
    python scripts/set_motor_id.py --new-id 1
    python scripts/set_motor_id.py --new-id 2
    ...
"""

import argparse
import glob
import os
import sys
import time
from typing import Optional

# Add root directory and DynamixelSDK to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../third_party/DynamixelSDK/python/src"
        )
    ),
)

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError:
    print("Error: DynamixelSDK not found.")
    sys.exit(1)

# Dynamixel Protocol 2.0 Control Table Addresses
ADDR_ID = 7
ADDR_BAUD_RATE = 8
ADDR_TORQUE_ENABLE = 64

# Baudrate code for Address 8: 1 = 57600 baud
BAUD_CODE_57600 = 1

COMMON_BAUDRATES = [57600, 1000000, 2000000, 115200]


def find_gello_port() -> Optional[str]:
    """Auto-detect USB serial port."""
    possible_ports = glob.glob("/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_*")
    if not possible_ports:
        possible_ports = glob.glob("/dev/ttyUSB*")

    if not possible_ports:
        return None
    elif len(possible_ports) == 1:
        return possible_ports[0]
    else:
        print("Multiple serial ports found:")
        for i, p in enumerate(possible_ports):
            print(f"  {i+1}: {p}")
        choice = input("Select port number (default 1): ").strip()
        idx = int(choice) - 1 if choice.isdigit() else 0
        return possible_ports[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Change Dynamixel motor ID for GELLO arms."
    )
    parser.add_argument(
        "--new-id",
        type=int,
        required=True,
        help="Target motor ID (1-252).",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="USB port path (e.g., /dev/ttyUSB0). If omitted, auto-detected.",
    )
    args = parser.parse_args()

    new_id = args.new_id
    if not (1 <= new_id <= 252):
        print(f"Error: Invalid ID {new_id}. ID must be between 1 and 252.")
        sys.exit(1)

    port_name = args.port or find_gello_port()
    if not port_name:
        print(
            "Error: Could not find any serial port (/dev/serial/by-id/* or /dev/ttyUSB*)."
        )
        print("Please make sure your U2D2 / USB converter is plugged in.")
        sys.exit(1)

    print(f"Connecting to port: {port_name}")

    port_handler = PortHandler(port_name)
    packet_handler = PacketHandler(2.0)

    if not port_handler.openPort():
        print(f"Error: Failed to open port {port_name}.")
        sys.exit(1)

    active_baud = None

    # Scan common baudrates to locate connected motor
    print("Scanning motor connection...")
    for baud in COMMON_BAUDRATES:
        port_handler.setBaudRate(baud)
        # Ping broadcast ID 254
        model_num, comm_res, err = packet_handler.ping(port_handler, 254)
        if comm_res == COMM_SUCCESS:
            active_baud = baud
            print(f"Found motor at baudrate {baud} (Model: {model_num})")
            break

    if active_baud is None:
        print("Could not detect motor via broadcast ping. Trying default baudrate 57600...")
        active_baud = 57600
        port_handler.setBaudRate(active_baud)

    # Step 1: Disable torque
    packet_handler.write1ByteTxRx(port_handler, 254, ADDR_TORQUE_ENABLE, 0)
    time.sleep(0.05)

    # Step 2: Set New ID
    print(f"Setting motor ID to {new_id}...")
    comm_res, err = packet_handler.write1ByteTxRx(port_handler, 254, ADDR_ID, new_id)
    if comm_res != COMM_SUCCESS:
        print(f"Warning: Write ID response: comm_res={comm_res}, err={err}. Retrying...")
        comm_res, err = packet_handler.write1ByteTxRx(
            port_handler, 254, ADDR_ID, new_id
        )

    time.sleep(0.05)

    # Step 3: Ensure Baud Rate is 57600
    if active_baud != 57600:
        print("Setting motor baud rate to 57600...")
        packet_handler.write1ByteTxRx(
            port_handler, new_id, ADDR_BAUD_RATE, BAUD_CODE_57600
        )
        time.sleep(0.05)
        port_handler.setBaudRate(57600)

    # Step 4: Verify connection with new ID
    model_num, comm_res, err = packet_handler.ping(port_handler, new_id)
    if comm_res == COMM_SUCCESS:
        print(
            f"SUCCESS: Motor ID changed to {new_id} at baudrate 57600! (Model: {model_num})"
        )
    else:
        print(f"Verification check: Ping to ID {new_id} returned code {comm_res}.")
        print(
            f"Motor ID update to {new_id} sent. Disconnect this motor and connect the next."
        )

    port_handler.closePort()


if __name__ == "__main__":
    main()

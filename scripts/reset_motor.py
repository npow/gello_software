"""Script to stop runaway spinning motors and factory reset Dynamixel servos.

Usage:
    Connect ONLY the spinning motor to the USB converter.
    python scripts/reset_motor.py
"""

import argparse
import glob
import os
import sys
import time

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

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# Control Table Addresses
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
POSITION_CONTROL_MODE = 3


def find_gello_port():
    possible_ports = glob.glob("/dev/serial/by-id/usb-FTDI_*") or glob.glob("/dev/ttyUSB*")
    return possible_ports[0] if possible_ports else None


def main():
    parser = argparse.ArgumentParser(
        description="Stop spinning Dynamixel motor and restore Position Control Mode."
    )
    parser.add_argument("--port", type=str, default=None, help="USB port path")
    args = parser.parse_args()

    port_name = args.port or find_gello_port()
    if not port_name:
        print("Error: No USB serial port found!")
        sys.exit(1)

    print(f"Connecting to port: {port_name}")
    port_handler = PortHandler(port_name)
    packet_handler = PacketHandler(2.0)

    if not port_handler.openPort():
        print(f"Failed to open port {port_name}")
        sys.exit(1)

    print("Attempting emergency torque disable & mode reset...")
    for baud in [57600, 1000000, 2000000, 115200]:
        port_handler.setBaudRate(baud)
        # Emergency Torque Disable via Broadcast ID 254
        packet_handler.write1ByteTxRx(port_handler, 254, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.05)
        # Reset Operating Mode to Position Control Mode (3)
        packet_handler.write1ByteTxRx(
            port_handler, 254, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE
        )
        time.sleep(0.05)

    print("Executing full Factory Reset...")
    for baud in [57600, 1000000]:
        port_handler.setBaudRate(baud)
        comm_res, err = packet_handler.factoryReset(port_handler, 254, 0xFF)
        if comm_res == COMM_SUCCESS:
            print("Factory Reset successful!")
            break

    time.sleep(0.5)

    # Re-apply 57600 baud rate and position mode at ID 1
    print("Re-configuring motor to ID 1, Baudrate 57600, Position Mode...")
    port_handler.setBaudRate(57600)
    packet_handler.write1ByteTxRx(port_handler, 1, ADDR_TORQUE_ENABLE, 0)
    packet_handler.write1ByteTxRx(
        port_handler, 1, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE
    )

    model, res, err = packet_handler.ping(port_handler, 1)
    if res == COMM_SUCCESS:
        print(
            f"SUCCESS: Motor is safely stopped, reset to ID 1, Position Control Mode, 57600 baud! (Model: {model})"
        )
    else:
        print(
            "Notice: Motor reset command sent. If motor continues spinning when powered, it may have a hardware failure (magnetic encoder / H-bridge short)."
        )

    port_handler.closePort()


if __name__ == "__main__":
    main()

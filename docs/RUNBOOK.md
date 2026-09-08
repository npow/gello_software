# YAM Physical Arms & GELLO Leaders: Operational Runbook & Crib Sheet

A fast-reference guide for launching, diagnosing, and resolving common hardware and communication failures with the bimanual I2RT YAM arms and GELLO leader controllers.

---

## 🚀 Quick Launch Sequence

Execute these steps in order every time you start a session:

```bash
# 1. Navigate to repository & activate virtual environment
cd /home/npow/code/gello_software
source .venv/bin/activate

# 2. Bring up / reset CAN interfaces (requires sudo)
sudo bash scripts/reset_all_can.sh

# 3. (Recommended) Run pre-flight hardware diagnostic
python scripts/check_hardware.py

# 4. Launch bimanual teleoperation
python experiments/launch_yaml.py \
  --left-config-path configs/yam_auto_generated.yaml \
  --right-config-path configs/yam_auto_generated_right.yaml
```

> [!TIP]
> **Data Collection**: To record demonstrations, append `--use-save-interface` to the launch command.
> - Press **`s`** to start saving a rollout.
> - Press **`q`** to stop saving and close the rollout.

---

## 🛠 Hardware Mapping Cheat Sheet

| Component | Left Side | Right Side |
| :--- | :--- | :--- |
| **Follower CAN Interface** | `can_left` | `can_right` |
| **CAN Bitrate** | 1,000,000 baud | 1,000,000 baud |
| **Follower Motor IDs** | 1 to 6 (Arm), 7 (Gripper) | 1 to 6 (Arm), 7 (Gripper) |
| **Follower Motor Type** | DM4310 / DM4340 / DM8009 | DM4310 / DM4340 / DM8009 |
| **GELLO Leader Serial** | `...FTBEQBRC-if00-port0` | `...FTBEQC10-if00-port0` |
| **Leader Baud Rate** | 57,600 baud | 57,600 baud |
| **Leader Motor IDs** | 1 to 6 (Arm), 7 (Gripper) | 1 to 6 (Arm), 7 (Gripper) |
| **Config File** | [`configs/yam_auto_generated.yaml`](file:///home/npow/code/gello_software/configs/yam_auto_generated.yaml) | [`configs/yam_auto_generated_right.yaml`](file:///home/npow/code/gello_software/configs/yam_auto_generated_right.yaml) |

---

## 🚒 Troubleshooting Matrix ("This Shit Happens All The Time")

### 1. `CanOperationError: Error receiving: Network is down [Error Code 100]`
- **Symptom**: Script crashes immediately when initializing CAN bus.
- **Cause**: SocketCAN interface `can_left` or `can_right` is down after boot or USB disconnect.
- **Fix**:
  ```bash
  sudo bash scripts/reset_all_can.sh
  ```
  Or bring them up manually:
  ```bash
  sudo ip link set can_left up type can bitrate 1000000
  sudo ip link set can_right up type can bitrate 1000000
  ```

---

### 2. `AssertionError: fail to communicate with the motor X on yam_real at can channel 'can_left'`
- **Symptom**: Motor 1..7 times out during `motor_on` handshake.
- **Root Cause**:
  1. The follower arm 24V/48V power supply is off or tripped.
  2. Motor `X` entered a latched fault state (most commonly `0xD` = *loss of communication* from a bus glitch, or `0x8`/`0x9` voltage error).
- **Fix**:
  1. **Run auto-recovery**:
     ```bash
     python scripts/check_hardware.py
     ```
     *(This queries each joint and automatically transmits the `[0xFF]*7 + [0xFB]` fault-clear frame).*
  2. **Power cycle**: Toggle off the 24V/48V DC power supply to the arm for 5 seconds, switch back on, then run:
     ```bash
     sudo bash scripts/reset_all_can.sh
     ```

---

### 3. `Failed to open port /dev/serial/by-id/...` or `Permission Denied` / Multiple Access
- **Symptom**: GELLO agent cannot open FTDI serial port, or reports `device reports readiness to read but returned no data (multiple access on port?)`.
- **Causes & Fixes**:
  - **Zombie Process**: When `launch_yaml.py` crashes *after* initializing one arm, non-daemon motor threads keep running in the background and hold the serial ports.
    Kill any orphaned processes:
    ```bash
    pkill -9 -f launch_yaml
    ```
  - **Permissions**: Ensure your user belongs to the `dialout` group:
    ```bash
    sudo usermod -aG dialout $USER
    # Requires logout / login if freshly added
    ```
  - **Unplugged / Port Swap**: Check which FTDI converters are plugged in:
    ```bash
    ls -l /dev/serial/by-id/usb-FTDI_*
    ```
  - If serial IDs changed, update [`configs/yam_auto_generated.yaml`](file:///home/npow/code/gello_software/configs/yam_auto_generated.yaml) or [`configs/yam_auto_generated_right.yaml`](file:///home/npow/code/gello_software/configs/yam_auto_generated_right.yaml).

---

### 4. GELLO Leader Runaway / Continuously Spinning Motor
- **Symptom**: A Dynamixel servo starts spinning infinitely or won't hold position mode.
- **Fix**:
  1. Unplug all other Dynamixels and plug **only** the runaway motor into the U2D2 adapter.
  2. Run the reset tool:
     ```bash
     python scripts/reset_motor.py
     ```
  3. Re-assign the proper joint ID:
     ```bash
     python scripts/set_motor_id.py --new-id <JOINT_ID>
     ```

---

### 5. Follower CAN Channels Swapped (Left acts as Right)
- **Symptom**: Moving the left leader moves the right arm, or vice versa.
- **Fix**: Verify udev rules in `/etc/udev/rules.d/90-can.rules`:
  ```bash
  udevadm info -a -p /sys/class/net/can* | grep -i serial
  ```
  Ensure the serial number for the left arm CAN adapter maps to `NAME="can_left"`, and the right arm maps to `NAME="can_right"`. Then reload:
  ```bash
  sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
  sudo bash scripts/reset_all_can.sh
  ```

---

### 6. Offsets Drifted or Arm Moves to Incorrect Pose
- **Symptom**: Leader zero position does not match follower zero position.
- **Fix**: Put the leader arm into the calibration pose and regenerate configs:
  ```bash
  # For left arm:
  python scripts/generate_yam_config.py --channel can_left --output-path configs/yam_auto_generated.yaml

  # For right arm:
  python scripts/generate_yam_config.py --channel can_right --output-path configs/yam_auto_generated_right.yaml
  ```

---

## ⚡ 10-Second Sanity Check Script

Run [`scripts/check_hardware.py`](file:///home/npow/code/gello_software/scripts/check_hardware.py) at any time:
```bash
python scripts/check_hardware.py
```
Outputs a full status checklist:
- CAN interfaces UP / DOWN
- All 14 follower joints (7 per arm) ping status & auto fault clearing
- All 14 leader Dynamixels (7 per arm) ping status

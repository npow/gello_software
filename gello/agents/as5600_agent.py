"""Serial agents for passive YAM leaders using encoders or potentiometers."""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

COUNTS_PER_REVOLUTION = 4096
RADIANS_PER_COUNT = 2.0 * math.pi / COUNTS_PER_REVOLUTION
POTENTIOMETER_COUNTS = 1024
POTENTIOMETER_RADIANS_PER_COUNT = math.radians(300.0) / (
    POTENTIOMETER_COUNTS - 1
)
PROTOCOL_COUNT_MAX = {"YAM1": 4095, "YAMP1": 1023}


class EncoderProtocolError(RuntimeError):
    """Raised when the encoder controller sends unsafe or malformed data."""


@dataclass(frozen=True)
class EncoderFrame:
    """One sample from the encoder controller."""

    counts: Tuple[int, ...]
    sequence: Optional[int] = None
    device_ms: Optional[int] = None
    deadman_pressed: Optional[bool] = None
    protocol_tag: Optional[str] = None


def parse_encoder_line(
    line: bytes, expected_channels: int = 7
) -> Optional[EncoderFrame]:
    """Parse a versioned YAM frame or the original comma-separated format.

    Versioned encoder (``YAM1``) and potentiometer (``YAMP1``) frames have the
    following fields::

        tag,sequence,device_ms,count0,...,count6,deadman

    Blank lines and diagnostic lines beginning with ``#`` are ignored. The
    legacy format is accepted to make bench testing with the original firmware
    convenient, but it does not carry a deadman or reboot counter.
    """

    stripped = line.strip()
    if not stripped or stripped.startswith(b"#"):
        return None

    try:
        text = stripped.decode("ascii")
    except UnicodeDecodeError as exc:
        raise EncoderProtocolError("Encoder frame is not ASCII") from exc

    fields = text.split(",")
    protocol_tag = fields[0] if fields[0] in PROTOCOL_COUNT_MAX else None
    if protocol_tag is not None:
        expected_fields = expected_channels + 4
        if len(fields) != expected_fields:
            raise EncoderProtocolError(
                f"Expected {expected_fields} fields in {protocol_tag} frame, "
                f"received {len(fields)}"
            )
        try:
            sequence = int(fields[1])
            device_ms = int(fields[2])
            counts = tuple(int(value) for value in fields[3:-1])
            deadman_raw = int(fields[-1])
        except ValueError as exc:
            raise EncoderProtocolError(
                "Encoder frame contains a non-integer field"
            ) from exc
        if deadman_raw not in (0, 1):
            raise EncoderProtocolError("Deadman field must be 0 or 1")
        frame = EncoderFrame(
            counts=counts,
            sequence=sequence,
            device_ms=device_ms,
            deadman_pressed=bool(deadman_raw),
            protocol_tag=protocol_tag,
        )
    else:
        if len(fields) != expected_channels:
            raise EncoderProtocolError(
                f"Expected {expected_channels} legacy encoder counts, "
                f"received {len(fields)} fields"
            )
        try:
            frame = EncoderFrame(counts=tuple(int(value) for value in fields))
        except ValueError as exc:
            raise EncoderProtocolError(
                "Encoder frame contains a non-integer count"
            ) from exc

    maximum_count = PROTOCOL_COUNT_MAX.get(
        frame.protocol_tag or "YAM1", COUNTS_PER_REVOLUTION - 1
    )
    bad_counts = [count for count in frame.counts if not 0 <= count <= maximum_count]
    if bad_counts:
        raise EncoderProtocolError(
            f"Encoder counts must be in [0, {maximum_count}], "
            f"received {bad_counts}"
        )
    return frame


class CountUnwrapper:
    """Track count deltas, optionally unwrapping a circular encoder."""

    def __init__(self, num_channels: int, count_modulus: Optional[int] = 4096):
        if count_modulus is not None and count_modulus <= 1:
            raise ValueError("count_modulus must be greater than one or None")
        self._last: Optional[np.ndarray] = None
        self._total = np.zeros(num_channels, dtype=np.int64)
        self._count_modulus = count_modulus

    def update(self, counts: Sequence[int]) -> np.ndarray:
        current = np.asarray(counts, dtype=np.int64)
        if current.shape != self._total.shape:
            raise ValueError(
                f"Expected {len(self._total)} encoder counts, received {len(current)}"
            )
        if self._last is None:
            self._last = current
            return self._total.copy()

        delta = current - self._last
        if self._count_modulus is not None:
            half_revolution = self._count_modulus // 2
            delta[delta > half_revolution] -= self._count_modulus
            delta[delta < -half_revolution] += self._count_modulus
        self._total += delta
        self._last = current
        return self._total.copy()


class EncoderMapper:
    """Map seven passive leader encoders into six YAM joints and a gripper."""

    def __init__(
        self,
        start_joints: Sequence[float],
        channel_map: Sequence[int],
        joint_signs: Sequence[float],
        joint_limits: Sequence[Sequence[float]],
        joint_scales: Optional[Sequence[float]] = None,
        gripper_joint_index: int = 6,
        gripper_travel_rad: float = 0.8,
        gripper_open_value: float = 1.0,
        gripper_closed_value: float = 0.0,
        radians_per_count: float = RADIANS_PER_COUNT,
        count_modulus: Optional[int] = COUNTS_PER_REVOLUTION,
    ):
        self._start_joints = np.asarray(start_joints, dtype=float)
        self._channel_map = np.asarray(channel_map, dtype=int)
        self._joint_signs = np.asarray(joint_signs, dtype=float)
        self._joint_limits = np.asarray(joint_limits, dtype=float)
        if joint_scales is None:
            self._joint_scales = np.ones(len(self._start_joints), dtype=float)
        else:
            self._joint_scales = np.asarray(joint_scales, dtype=float)
        self._gripper_joint_index = gripper_joint_index
        self._gripper_travel_rad = float(gripper_travel_rad)
        self._gripper_open_value = float(gripper_open_value)
        self._gripper_closed_value = float(gripper_closed_value)
        self._radians_per_count = float(radians_per_count)

        num_joints = len(self._start_joints)
        if num_joints != 7:
            raise ValueError(
                f"A YAM encoder leader requires 7 outputs, received {num_joints}"
            )
        for name, values in (
            ("channel_map", self._channel_map),
            ("joint_signs", self._joint_signs),
            ("joint_scales", self._joint_scales),
        ):
            if values.shape != (num_joints,):
                raise ValueError(f"{name} must have {num_joints} entries")
        if sorted(self._channel_map.tolist()) != list(range(num_joints)):
            raise ValueError(
                "channel_map must be a permutation of encoder channels 0 through 6"
            )
        if not np.all(np.abs(self._joint_signs) == 1):
            raise ValueError("joint_signs entries must be either -1 or 1")
        if np.any(self._joint_scales <= 0):
            raise ValueError("joint_scales entries must be positive")
        if self._joint_limits.shape != (num_joints, 2):
            raise ValueError(
                "joint_limits must contain [lower, upper] for all 7 outputs"
            )
        if np.any(self._joint_limits[:, 0] >= self._joint_limits[:, 1]):
            raise ValueError("Each lower joint limit must be below its upper limit")
        if np.any(self._start_joints < self._joint_limits[:, 0]) or np.any(
            self._start_joints > self._joint_limits[:, 1]
        ):
            raise ValueError("start_joints must be inside the configured joint limits")
        if not 0 <= gripper_joint_index < num_joints:
            raise ValueError("gripper_joint_index is out of range")
        if self._gripper_travel_rad <= 0:
            raise ValueError("gripper_travel_rad must be positive")
        if self._radians_per_count <= 0:
            raise ValueError("radians_per_count must be positive")
        gripper_limits = self._joint_limits[gripper_joint_index]
        for name, value in (
            ("gripper_open_value", self._gripper_open_value),
            ("gripper_closed_value", self._gripper_closed_value),
        ):
            if not gripper_limits[0] <= value <= gripper_limits[1]:
                raise ValueError(f"{name} must be inside the gripper joint limits")

        self._unwrapper = CountUnwrapper(num_joints, count_modulus=count_modulus)
        self._calibrated = False

    def calibrate(self, counts: Sequence[int]) -> np.ndarray:
        """Use the current encoder readings as the configured YAM start pose."""

        self._unwrapper.update(counts)
        self._calibrated = True
        return self._start_joints.copy()

    def map_counts(self, counts: Sequence[int]) -> np.ndarray:
        """Convert a raw encoder sample to a bounded YAM command."""

        if not self._calibrated:
            raise RuntimeError("EncoderMapper must be calibrated before mapping counts")

        unwrapped = self._unwrapper.update(counts)
        encoder_delta_rad = unwrapped * self._radians_per_count
        joint_delta_rad = (
            encoder_delta_rad[self._channel_map]
            * self._joint_signs
            * self._joint_scales
        )
        command = self._start_joints + joint_delta_rad

        gripper_progress = np.clip(
            joint_delta_rad[self._gripper_joint_index] / self._gripper_travel_rad,
            0.0,
            1.0,
        )
        command[self._gripper_joint_index] = self._gripper_open_value + (
            gripper_progress * (self._gripper_closed_value - self._gripper_open_value)
        )
        return np.clip(command, self._joint_limits[:, 0], self._joint_limits[:, 1])


class AS5600SerialAgent:
    """Read a seven-channel ESP32 encoder leader and command a YAM follower.

    The leader must be held in ``start_joints`` when this class connects. The
    first complete frame becomes the calibration sample, so magnet mounting
    angle does not need to be measured during assembly.
    """

    def __init__(
        self,
        port: str,
        start_joints: Sequence[float],
        joint_limits: Sequence[Sequence[float]],
        channel_map: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
        joint_signs: Sequence[float] = (1, -1, -1, -1, 1, 1, 1),
        joint_scales: Optional[Sequence[float]] = None,
        gripper_joint_index: int = 6,
        gripper_travel_rad: float = 0.8,
        gripper_open_value: float = 1.0,
        gripper_closed_value: float = 0.0,
        baudrate: int = 115200,
        serial_read_timeout_s: float = 0.02,
        startup_timeout_s: float = 5.0,
        frame_timeout_s: float = 0.25,
        require_deadman: bool = True,
        max_output_rates: Optional[Sequence[float]] = None,
        serial_connection: Optional[Any] = None,
        radians_per_count: float = RADIANS_PER_COUNT,
        count_modulus: Optional[int] = COUNTS_PER_REVOLUTION,
        expected_protocol_tag: Optional[str] = "YAM1",
    ):
        self._num_channels = len(start_joints)
        self._frame_timeout_s = float(frame_timeout_s)
        self._require_deadman = require_deadman
        self._expected_protocol_tag = expected_protocol_tag
        if self._frame_timeout_s <= 0 or startup_timeout_s <= 0:
            raise ValueError("Serial timeouts must be positive")

        self._mapper = EncoderMapper(
            start_joints=start_joints,
            channel_map=channel_map,
            joint_signs=joint_signs,
            joint_limits=joint_limits,
            joint_scales=joint_scales,
            gripper_joint_index=gripper_joint_index,
            gripper_travel_rad=gripper_travel_rad,
            gripper_open_value=gripper_open_value,
            gripper_closed_value=gripper_closed_value,
            radians_per_count=radians_per_count,
            count_modulus=count_modulus,
        )

        if max_output_rates is None:
            self._max_output_rates = None
        else:
            self._max_output_rates = np.asarray(max_output_rates, dtype=float)
            if self._max_output_rates.shape != (self._num_channels,):
                raise ValueError("max_output_rates must have one entry per output")
            if np.any(self._max_output_rates <= 0):
                raise ValueError("max_output_rates entries must be positive")

        if serial_connection is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError(
                    "pyserial is required for AS5600 leaders; install requirements.txt"
                ) from exc
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=serial_read_timeout_s,
            )
        else:
            self._serial = serial_connection

        self._last_sequence: Optional[int] = None
        self._last_action: Optional[np.ndarray] = None
        self._last_action_time: Optional[float] = None

        print(
            "Hold the encoder leader in the configured YAM start pose; "
            "waiting for a complete frame..."
        )
        first_frame = self._read_frame(startup_timeout_s)
        self._validate_sequence(first_frame)
        self._last_action = self._mapper.calibrate(first_frame.counts)
        self._last_action_time = time.monotonic()
        print(f"Passive leader calibrated on counts {list(first_frame.counts)}")

    def _read_frame(self, timeout_s: float) -> EncoderFrame:
        deadline = time.monotonic() + timeout_s
        latest: Optional[EncoderFrame] = None
        while time.monotonic() < deadline:
            line = self._serial.readline()
            if line:
                frame = parse_encoder_line(line, self._num_channels)
                if frame is not None:
                    self._validate_protocol(frame)
                    latest = frame

                # Drain buffered samples so the follower uses the newest pose.
                while getattr(self._serial, "in_waiting", 0):
                    buffered = self._serial.readline()
                    if not buffered:
                        break
                    frame = parse_encoder_line(buffered, self._num_channels)
                    if frame is not None:
                        self._validate_protocol(frame)
                        latest = frame
                if latest is not None:
                    return latest
        raise EncoderProtocolError(
            f"No complete encoder frame received for {timeout_s:.3f} seconds"
        )

    def _validate_protocol(self, frame: EncoderFrame) -> None:
        if self._expected_protocol_tag is None:
            return
        if frame.protocol_tag != self._expected_protocol_tag:
            received = frame.protocol_tag or "legacy unversioned"
            raise EncoderProtocolError(
                f"Expected {self._expected_protocol_tag} firmware, received {received}"
            )

    def _validate_sequence(self, frame: EncoderFrame) -> None:
        if frame.sequence is None:
            if self._require_deadman:
                raise EncoderProtocolError(
                    "Deadman safety requires versioned firmware frames"
                )
            return
        if self._last_sequence is not None:
            sequence_delta = (frame.sequence - self._last_sequence) & 0xFFFFFFFF
            if sequence_delta == 0 or sequence_delta >= 0x80000000:
                raise EncoderProtocolError(
                    "Encoder firmware sequence moved backward or restarted; recalibrate"
                )
        self._last_sequence = frame.sequence

    def _rate_limit(self, target: np.ndarray, now: float) -> np.ndarray:
        if (
            self._max_output_rates is None
            or self._last_action is None
            or self._last_action_time is None
        ):
            return target
        elapsed = max(now - self._last_action_time, 0.0)
        allowed_delta = self._max_output_rates * elapsed
        return self._last_action + np.clip(
            target - self._last_action, -allowed_delta, allowed_delta
        )

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Return the latest safe leader pose for the GELLO control loop."""

        frame = self._read_frame(self._frame_timeout_s)
        self._validate_sequence(frame)
        now = time.monotonic()
        # Keep rollover tracking current even while the follower is held by the
        # deadman. This prevents a wrap during a pause from looking like a jump.
        target = self._mapper.map_counts(frame.counts)

        if self._require_deadman:
            if frame.deadman_pressed is None:
                raise EncoderProtocolError("Firmware frame is missing the deadman state")
            if not frame.deadman_pressed:
                follower_joints = np.asarray(obs["joint_positions"], dtype=float)
                if follower_joints.shape != (self._num_channels,):
                    raise EncoderProtocolError(
                        "Follower observation shape does not match encoder leader output"
                    )
                self._last_action = follower_joints.copy()
                self._last_action_time = now
                return follower_joints

        action = self._rate_limit(target, now)
        self._last_action = action
        self._last_action_time = now
        return action.copy()

    def close(self) -> None:
        """Close the serial device."""

        if hasattr(self._serial, "close"):
            self._serial.close()


class PotentiometerSerialAgent(AS5600SerialAgent):
    """Read the no-solder seven-potentiometer Nano leader."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("radians_per_count", POTENTIOMETER_RADIANS_PER_COUNT)
        kwargs.setdefault("count_modulus", None)
        kwargs.setdefault("expected_protocol_tag", "YAMP1")
        super().__init__(*args, **kwargs)

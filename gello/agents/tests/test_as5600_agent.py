import math

import numpy as np
import pytest

from gello.agents.as5600_agent import (
    AS5600SerialAgent,
    CountUnwrapper,
    EncoderMapper,
    EncoderProtocolError,
    parse_encoder_line,
)

YAM_LIMITS = [
    [-5 * math.pi / 6, math.pi],
    [0.0, 7 * math.pi / 6],
    [0.0, math.pi],
    [-1.69297, math.pi / 2],
    [-math.pi / 2, math.pi / 2],
    [-2 * math.pi / 3, 2 * math.pi / 3],
    [0.0, 1.0],
]


def test_parse_versioned_frame():
    frame = parse_encoder_line(b"YAM1,42,1234,0,1,2,3,4,5,4095,1\n")

    assert frame is not None
    assert frame.sequence == 42
    assert frame.device_ms == 1234
    assert frame.counts == (0, 1, 2, 3, 4, 5, 4095)
    assert frame.deadman_pressed is True


def test_parse_rejects_failed_sensor():
    with pytest.raises(EncoderProtocolError, match="counts must be"):
        parse_encoder_line(b"YAM1,1,10,0,1,2,-1,4,5,6,1\n")


def test_count_unwrapper_crosses_zero_both_directions():
    unwrapper = CountUnwrapper(2)

    np.testing.assert_array_equal(unwrapper.update([4090, 5]), [0, 0])
    np.testing.assert_array_equal(unwrapper.update([3, 4092]), [9, -9])


def test_mapper_applies_channel_sign_gripper_and_limits():
    mapper = EncoderMapper(
        start_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        channel_map=[1, 0, 2, 3, 4, 5, 6],
        joint_signs=[1, -1, 1, 1, 1, 1, 1],
        joint_limits=YAM_LIMITS,
        gripper_travel_rad=math.pi / 2,
    )
    mapper.calibrate([1000] * 7)

    command = mapper.map_counts(
        [1000 + 512, 1000 + 256, 1000, 1000, 1000, 1000, 1000 + 512]
    )

    assert command[0] == pytest.approx(math.pi / 8)
    # Joint 2 cannot move below its zero lower limit, so the signed delta clamps.
    assert command[1] == 0.0
    assert command[6] == pytest.approx(0.5)


class FakeSerial:
    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    @property
    def in_waiting(self):
        # Expose one frame at a time. Draining behavior is a pyserial concern;
        # these tests exercise calibration followed by a later control sample.
        return 0

    def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def close(self):
        self.closed = True


def test_agent_holds_follower_while_deadman_is_released():
    serial = FakeSerial(
        [
            b"YAM1,1,10,100,100,100,100,100,100,100,0\n",
            b"YAM1,2,20,200,200,200,200,200,200,200,0\n",
        ]
    )
    agent = AS5600SerialAgent(
        port="unused",
        start_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        joint_limits=YAM_LIMITS,
        serial_connection=serial,
        require_deadman=True,
    )
    follower = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    action = agent.act({"joint_positions": follower})

    np.testing.assert_array_equal(action, follower)


def test_agent_rejects_firmware_restart():
    serial = FakeSerial(
        [
            b"YAM1,10,100,100,100,100,100,100,100,100,1\n",
            b"YAM1,1,5,101,101,101,101,101,101,101,1\n",
        ]
    )
    agent = AS5600SerialAgent(
        port="unused",
        start_joints=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        joint_limits=YAM_LIMITS,
        serial_connection=serial,
        require_deadman=True,
    )

    with pytest.raises(EncoderProtocolError, match="restarted"):
        agent.act({"joint_positions": np.zeros(7)})

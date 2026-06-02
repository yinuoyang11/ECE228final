import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_libero_rollout.py"
SPEC = importlib.util.spec_from_file_location("eval_libero_rollout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROLLOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROLLOUT)


def test_quaternion_to_axis_angle_identity():
    axis_angle = ROLLOUT.quaternion_to_axis_angle(np.array([0.0, 0.0, 0.0, 1.0]))

    assert np.allclose(axis_angle, np.zeros(3))


def test_quaternion_to_axis_angle_half_turn():
    axis_angle = ROLLOUT.quaternion_to_axis_angle(np.array([1.0, 0.0, 0.0, 0.0]))

    assert np.allclose(axis_angle, np.array([np.pi, 0.0, 0.0]), atol=1e-6)


def test_rotate_image_flips_height_and_width():
    image = np.arange(12).reshape(2, 2, 3)

    assert np.array_equal(ROLLOUT.rotate_image(image), image[::-1, ::-1])

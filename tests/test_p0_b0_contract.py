from collections.abc import Mapping
import unittest

import torch

from src.ppo_stackcube import DictArray, NatureCNN, restrict_policy_observation


class StateAccessTrap(Mapping):
    """Mapping that fails if P0 ever tries to retrieve state."""

    def __init__(self, rgb):
        self.rgb = rgb

    def __getitem__(self, key):
        if key == "rgb":
            return self.rgb
        if key == "state":
            raise AssertionError("P0 accessed state")
        raise KeyError(key)

    def __iter__(self):
        return iter(("rgb", "state"))

    def __len__(self):
        return 2


class P0B0ObservationContractTest(unittest.TestCase):
    def setUp(self):
        self.rgb = torch.zeros((3, 64, 64, 6), dtype=torch.uint8)

    def test_p0_does_not_access_or_return_state(self):
        restricted = restrict_policy_observation(StateAccessTrap(self.rgb), "p0")
        self.assertEqual(set(restricted), {"rgb"})
        self.assertIs(restricted["rgb"], self.rgb)

    def test_b0_keeps_exactly_the_first_25_state_dimensions(self):
        state = torch.randn((3, 48))
        restricted = restrict_policy_observation(
            {"rgb": self.rgb, "state": state}, "b0"
        )
        self.assertEqual(tuple(restricted["state"].shape), (3, 25))
        torch.testing.assert_close(restricted["state"], state[..., :25])

    def test_privileged_dimensions_cannot_change_b0_input(self):
        state_a = torch.randn((3, 48))
        state_b = state_a.clone()
        state_b[..., 25:] = torch.randn_like(state_b[..., 25:]) * 1000
        input_a = restrict_policy_observation(
            {"rgb": self.rgb, "state": state_a}, "b0"
        )
        input_b = restrict_policy_observation(
            {"rgb": self.rgb, "state": state_b}, "b0"
        )
        torch.testing.assert_close(input_a["state"], input_b["state"])

    def test_rollout_buffers_have_no_privileged_state_capacity(self):
        p0_obs = restrict_policy_observation(StateAccessTrap(self.rgb), "p0")
        p0_buffer = DictArray((2, 3), p0_obs)
        self.assertEqual(set(p0_buffer.keys()), {"rgb"})

        state = torch.randn((3, 48))
        b0_obs = restrict_policy_observation(
            {"rgb": self.rgb, "state": state}, "b0"
        )
        b0_buffer = DictArray((2, 3), b0_obs)
        self.assertEqual(tuple(b0_buffer["state"].shape), (2, 3, 25))

    def test_p0_and_b0_share_the_same_rgb_cnn(self):
        p0 = NatureCNN({"rgb": self.rgb})
        b0 = NatureCNN(
            {"rgb": self.rgb, "state": torch.zeros((3, 25))}
        )
        p0_shapes = [tuple(p.shape) for p in p0.extractors["rgb"].parameters()]
        b0_shapes = [tuple(p.shape) for p in b0.extractors["rgb"].parameters()]
        self.assertEqual(p0_shapes, b0_shapes)

    def test_b0_rejects_unexpected_state_layout(self):
        with self.assertRaisesRegex(AssertionError, "48-D"):
            restrict_policy_observation(
                {"rgb": self.rgb, "state": torch.zeros((3, 47))}, "b0"
            )


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from src.ppo_stackcube import RunningMeanStd, restrict_policy_observation


class B0NormContractTest(unittest.TestCase):
    def test_b0_norm_keeps_only_proprio(self):
        rgb = torch.zeros((4, 64, 64, 6), dtype=torch.uint8)
        state = torch.randn((4, 48))
        restricted = restrict_policy_observation(
            {"rgb": rgb, "state": state}, "b0_norm"
        )
        self.assertEqual(tuple(restricted["state"].shape), (4, 25))
        torch.testing.assert_close(restricted["state"], state[:, :25])

    def test_running_statistics_match_concatenated_population(self):
        first = torch.tensor([[1.0, 10.0], [3.0, 14.0]])
        second = torch.tensor([[5.0, 18.0], [7.0, 22.0]])
        rms = RunningMeanStd(2, epsilon=0.0)
        rms.update(first)
        rms.update(second)
        all_values = torch.cat([first, second], dim=0).to(torch.float64)
        torch.testing.assert_close(rms.mean, all_values.mean(dim=0))
        torch.testing.assert_close(
            rms.var, all_values.var(dim=0, unbiased=False)
        )
        self.assertEqual(rms.count.item(), 4.0)

    def test_normalization_is_featurewise_and_clipped(self):
        rms = RunningMeanStd(2, epsilon=0.0)
        rms.update(torch.tensor([[-1.0, 8.0], [1.0, 12.0]]))
        normalized = rms.normalize(torch.tensor([[0.0, 100.0]]), clip=2.0)
        torch.testing.assert_close(normalized, torch.tensor([[0.0, 2.0]]))

    def test_statistics_survive_checkpoint_roundtrip(self):
        source = RunningMeanStd(3)
        source.update(torch.randn((32, 3)))
        restored = RunningMeanStd(3)
        restored.load_state_dict(source.state_dict())
        torch.testing.assert_close(restored.mean, source.mean)
        torch.testing.assert_close(restored.var, source.var)
        torch.testing.assert_close(restored.count, source.count)


if __name__ == "__main__":
    unittest.main()

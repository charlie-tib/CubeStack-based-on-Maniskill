from io import BytesIO
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.normal import Normal

from src.ppo_stackcube import (
    AblationAgent,
    Args,
    gaussian_policy_kl,
    pre_success_caps_transition_mask,
    rollout_action_diagnostics,
    shared_rgb_photometric_augmentation,
    temporal_caps_loss,
)


class DummyEnv:
    def __init__(self):
        low = np.array([-1.0, -0.5, -2.0, -1.0, -1.0, -0.25, -1.0, -1.0, -0.75])
        high = np.array([1.0, 0.75, 2.0, 1.0, 1.0, 0.50, 1.0, 1.0, 0.25])
        self.unwrapped = SimpleNamespace(
            single_action_space=gym.spaces.Box(
                low=low.astype(np.float32),
                high=high.astype(np.float32),
                dtype=np.float32,
            )
        )


def make_args(policy_distribution="gaussian"):
    args = Args(
        experiment="b4_norm",
        proprio_dim=25,
        frame_stack=4,
        policy_distribution=policy_distribution,
    )
    return args


def make_obs(batch_size=8):
    return {
        "rgb": torch.randint(
            0, 256, (batch_size, 64, 64, 24), dtype=torch.uint8
        ),
        "state": torch.randn(batch_size, 48),
    }


class PPOOptimizationContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.env = DummyEnv()
        self.obs = make_obs()

    def test_default_gaussian_path_is_elementwise_legacy_equivalent(self):
        agent = AblationAgent(self.env, self.obs, make_args())
        self.assertEqual(agent.policy_distribution, "gaussian")
        self.assertTrue(hasattr(agent, "actor_logstd"))
        self.assertFalse(hasattr(agent, "action_low"))

        torch.manual_seed(91)
        actual = agent.get_action_and_value(self.obs)
        torch.manual_seed(91)
        actor_features, visual_features = agent.actor_encoder.encode(self.obs)
        mean = agent.actor_mean(actor_features)
        logstd = agent.actor_logstd.expand_as(mean)
        probs = Normal(mean, torch.exp(logstd))
        legacy_action = probs.sample()
        expected = (
            legacy_action,
            probs.log_prob(legacy_action).sum(1),
            probs.entropy().sum(1),
            agent.critic(agent.critic_encoder(self.obs)),
            agent.aux_head(visual_features),
        )
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_beta_policy_is_bounded_finite_and_checkpoint_roundtrips(self):
        agent = AblationAgent(self.env, self.obs, make_args("beta"))
        self.assertFalse(hasattr(agent, "actor_logstd"))
        with torch.no_grad():
            features, _ = agent.actor_encoder.encode(self.obs)
            beta_distribution = agent._beta_distribution_from_features(features)
        self.assertAlmostEqual(
            beta_distribution.concentration1.mean().item(), 1.01, places=3
        )
        self.assertAlmostEqual(
            beta_distribution.concentration0.mean().item(), 1.01, places=3
        )
        low = torch.as_tensor(self.env.unwrapped.single_action_space.low)
        high = torch.as_tensor(self.env.unwrapped.single_action_space.high)

        action, logprob, entropy, value, aux = agent.get_action_and_value(self.obs)
        self.assertTrue(torch.all(action >= low))
        self.assertTrue(torch.all(action <= high))
        for tensor in (action, logprob, entropy, value, aux):
            self.assertTrue(torch.isfinite(tensor).all())

        deterministic = agent.get_action(self.obs, deterministic=True)
        torch.testing.assert_close(deterministic, agent.get_action_mean(self.obs))
        self.assertTrue(torch.all(deterministic >= low))
        self.assertTrue(torch.all(deterministic <= high))

        _, replay_logprob, replay_entropy, _, _ = agent.get_action_and_value(
            self.obs, action
        )
        torch.testing.assert_close(replay_logprob, logprob)
        self.assertTrue(torch.isfinite(replay_entropy).all())

        payload = BytesIO()
        torch.save(agent.state_dict(), payload)
        payload.seek(0)
        restored = AblationAgent(self.env, self.obs, make_args("beta"))
        restored.load_state_dict(torch.load(payload, weights_only=True))
        torch.testing.assert_close(
            restored.get_action(self.obs, deterministic=True), deterministic
        )

    def test_temporal_caps_excludes_reset_boundaries(self):
        current = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        following = torch.tensor([[2.0, 4.0], [1002.0, -998.0]])
        valid = torch.tensor([True, False])
        action_range = torch.tensor([2.0, 4.0])
        loss = temporal_caps_loss(current, following, valid, action_range)
        self.assertEqual(loss.item(), 1.0)

        no_pairs = temporal_caps_loss(
            current.requires_grad_(), following, torch.zeros(2, dtype=torch.bool),
            action_range,
        )
        self.assertEqual(no_pairs.item(), 0.0)
        no_pairs.backward()
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))

    def test_pre_success_caps_mask_stays_off_until_episode_reset(self):
        success_seen = torch.tensor([False, False])

        valid, success_seen = pre_success_caps_transition_mask(
            success_seen,
            torch.tensor([False, True]),
            torch.tensor([False, False]),
        )
        torch.testing.assert_close(valid, torch.tensor([True, False]))
        torch.testing.assert_close(success_seen, torch.tensor([False, True]))

        valid, success_seen = pre_success_caps_transition_mask(
            success_seen,
            torch.tensor([True, False]),
            torch.tensor([False, False]),
        )
        torch.testing.assert_close(valid, torch.tensor([False, False]))
        torch.testing.assert_close(success_seen, torch.tensor([True, True]))

        valid, success_seen = pre_success_caps_transition_mask(
            success_seen,
            torch.tensor([False, False]),
            torch.tensor([True, False]),
        )
        torch.testing.assert_close(valid, torch.tensor([False, False]))
        torch.testing.assert_close(success_seen, torch.tensor([False, True]))

        valid, success_seen = pre_success_caps_transition_mask(
            success_seen,
            torch.tensor([False, False]),
            torch.tensor([False, True]),
        )
        torch.testing.assert_close(valid, torch.tensor([True, False]))
        torch.testing.assert_close(success_seen, torch.tensor([False, False]))

    def test_photometric_aug_changes_only_rgb_and_shares_camera_frame_jitter(self):
        constant_obs = {
            "rgb": torch.full((4, 16, 16, 24), 128, dtype=torch.uint8),
            "state": torch.randn(4, 48),
        }
        torch.manual_seed(3)
        augmented = shared_rgb_photometric_augmentation(constant_obs)
        self.assertIs(augmented["state"], constant_obs["state"])
        self.assertEqual(augmented["rgb"].dtype, constant_obs["rgb"].dtype)
        self.assertEqual(augmented["rgb"].shape, constant_obs["rgb"].shape)
        self.assertFalse(torch.equal(augmented["rgb"], constant_obs["rgb"]))
        for sample in augmented["rgb"]:
            self.assertEqual(torch.unique(sample).numel(), 1)

        agent = AblationAgent(self.env, make_obs(4), make_args())
        original = make_obs(4)
        torch.manual_seed(4)
        changed = shared_rgb_photometric_augmentation(original)
        torch.testing.assert_close(agent.get_value(original), agent.get_value(changed))
        self.assertIs(changed["state"], original["state"])

    def test_gaussian_kl_and_rollout_diagnostics_are_finite(self):
        mean = torch.zeros(3, 9)
        logstd = torch.full_like(mean, -0.5)
        self.assertEqual(
            gaussian_policy_kl(mean, logstd, mean, logstd).item(), 0.0
        )
        shifted = mean + 0.2
        kl = gaussian_policy_kl(mean, logstd, shifted, logstd)
        self.assertGreater(kl.item(), 0.0)
        self.assertTrue(torch.isfinite(kl))

        actions = torch.zeros(3, 2, 9)
        actions[1, 0] = 100.0
        dones = torch.zeros(3, 2)
        dones[1, 0] = 1.0
        low = torch.as_tensor(self.env.unwrapped.single_action_space.low)
        high = torch.as_tensor(self.env.unwrapped.single_action_space.high)
        diagnostics = rollout_action_diagnostics(actions, dones, low, high)
        self.assertGreater(diagnostics["action_out_of_bounds_fraction"].item(), 0)
        self.assertTrue(
            all(torch.isfinite(value) for value in diagnostics.values())
        )


if __name__ == "__main__":
    unittest.main()

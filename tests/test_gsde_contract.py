from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Normal

from src.ppo_gsde import (
    Agent,
    action_delta_statistics,
    should_resample_sde,
)


class DummyEnv:
    def __init__(self):
        self.unwrapped = SimpleNamespace(
            single_action_space=gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(9,),
                dtype=np.float32,
            )
        )


def make_obs(batch_size=4, identical=False):
    rgb = torch.randint(
        0, 256, (batch_size, 64, 64, 6), dtype=torch.uint8
    )
    state = torch.randn(batch_size, 25)
    if identical:
        rgb = rgb[:1].expand(batch_size, -1, -1, -1).clone()
        state = state[:1].expand(batch_size, -1).clone()
    return {"rgb": rgb, "state": state}


def load_original_agent_class():
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "ppo_stackcube.py"
    )
    spec = spec_from_file_location("original_b0_norm", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


class GsdeContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        self.env = DummyEnv()

    def test_non_gsde_control_matches_original_b0_actor_and_critic(self):
        original_agent_class = load_original_agent_class()
        obs = make_obs(batch_size=3)

        torch.manual_seed(123)
        original = original_agent_class(
            self.env, obs, normalize_state=True
        )
        torch.manual_seed(123)
        control = Agent(
            self.env, obs, normalize_state=True, use_gsde=False
        )

        self.assertEqual(
            sum(p.numel() for p in original.parameters()),
            sum(p.numel() for p in control.parameters()),
        )
        with torch.no_grad():
            original_action = original.get_action(obs, deterministic=True)
            control_action = control.get_action(obs, deterministic=True)
            original_value = original.get_value(obs)
            control_value = control.get_value(obs)
        torch.testing.assert_close(control_action, original_action)
        torch.testing.assert_close(control_value, original_value)
        torch.testing.assert_close(
            control.actor_logstd, original.actor_logstd
        )

    def test_gsde_uses_full_feature_by_action_noise_matrix(self):
        obs = make_obs(batch_size=4, identical=True)
        agent = Agent(
            self.env,
            obs,
            normalize_state=True,
            use_gsde=True,
            sde_log_std_init=-2.0,
        )
        self.assertEqual(tuple(agent.actor_logstd.shape), (512, 9))

        agent.reset_noise(4)
        self.assertEqual(
            tuple(agent._sde_exploration_matrices.shape),
            (4, 512, 9),
        )
        with torch.no_grad():
            action_a = agent.get_action(obs)
            action_b = agent.get_action(obs)
            deterministic_a = agent.get_action(obs, deterministic=True)
            agent.reset_noise(4)
            deterministic_b = agent.get_action(obs, deterministic=True)
            action_c = agent.get_action(obs)

        # The same sampled matrix produces consistent noise over time.
        torch.testing.assert_close(action_a, action_b)
        # Each parallel environment receives its own exploration matrix.
        self.assertFalse(torch.equal(action_a[0], action_a[1]))
        # Evaluation is the policy mean and does not depend on gSDE noise.
        torch.testing.assert_close(deterministic_a, deterministic_b)
        # Resampling changes stochastic actions.
        self.assertFalse(torch.equal(action_a, action_c))

    def test_analytic_marginal_std_and_log_prob_match_equation(self):
        obs = make_obs(batch_size=3)
        agent = Agent(
            self.env,
            obs,
            normalize_state=True,
            use_gsde=True,
            sde_log_std_init=-2.0,
            sde_epsilon=1e-6,
        )
        with torch.no_grad():
            features = agent.feature_net(obs)
            latent = agent.actor_latent(features)
            mean = agent.actor_mean(latent)
            expected_std = torch.sqrt(
                latent.square()
                @ torch.exp(agent.actor_logstd).square()
                + agent.sde_epsilon
            )
            actual_std = agent.get_action_std(obs)
            agent.reset_noise(3)
            action = agent.get_action(obs)
        torch.testing.assert_close(actual_std, expected_std)

        _, log_prob, entropy, _, _ = agent.get_action_and_value(
            obs, action=action
        )
        expected_distribution = Normal(mean, expected_std)
        torch.testing.assert_close(
            log_prob,
            expected_distribution.log_prob(action).sum(1),
        )
        torch.testing.assert_close(
            entropy,
            expected_distribution.entropy().sum(1),
        )

    def test_ppo_log_prob_updates_gsde_variance(self):
        obs = make_obs(batch_size=4)
        agent = Agent(
            self.env,
            obs,
            normalize_state=True,
            use_gsde=True,
        )
        with torch.no_grad():
            agent.reset_noise(4)
            action = agent.get_action(obs)
        _, log_prob, _, _, _ = agent.get_action_and_value(
            obs, action=action
        )
        (-log_prob.mean()).backward()
        gradient = agent.actor_logstd.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_initial_effective_std_can_match_gaussian_baseline(self):
        obs = make_obs(batch_size=8)
        target_std = float(np.exp(-0.5))
        agent = Agent(
            self.env,
            obs,
            normalize_state=True,
            use_gsde=True,
            sde_log_std_init=-2.0,
        )
        initial_std = agent.get_action_std(obs).mean().item()
        agent.calibrate_sde_action_std(obs, target_std)
        calibrated_std = agent.get_action_std(obs).mean().item()

        self.assertGreater(initial_std, target_std)
        self.assertAlmostEqual(calibrated_std, target_std, places=5)

    def test_gsde_checkpoint_roundtrip_preserves_policy_and_variance(self):
        obs = make_obs(batch_size=4)
        agent = Agent(
            self.env,
            obs,
            normalize_state=True,
            use_gsde=True,
        )
        agent.calibrate_sde_action_std(obs, float(np.exp(-0.5)))
        clone = Agent(
            self.env,
            obs,
            normalize_state=True,
            use_gsde=True,
        )
        clone.load_state_dict(agent.state_dict())

        with torch.no_grad():
            torch.testing.assert_close(
                clone.get_action(obs, deterministic=True),
                agent.get_action(obs, deterministic=True),
            )
            torch.testing.assert_close(
                clone.get_action_std(obs), agent.get_action_std(obs)
            )
            torch.testing.assert_close(
                clone.actor_logstd, agent.actor_logstd
            )

    def test_resampling_schedule_and_continuity_cost(self):
        expected = [True, False, False, False, True, False, False, False]
        actual = [should_resample_sde(step, 4) for step in range(8)]
        self.assertEqual(actual, expected)
        self.assertFalse(any(should_resample_sde(step, -1) for step in range(8)))

        previous = -torch.ones(2, 9)
        action = torch.ones(2, 9)
        valid = torch.tensor([True, False])
        action_range = torch.full((9,), 2.0)
        square_sum, absolute_sum, count = action_delta_statistics(
            action, previous, valid, action_range
        )
        self.assertEqual(count, 9)
        self.assertEqual(square_sum.item(), 9.0)
        self.assertEqual(absolute_sum.item(), 9.0)
        self.assertEqual(100.0 * square_sum.item() / count, 100.0)


if __name__ == "__main__":
    unittest.main()

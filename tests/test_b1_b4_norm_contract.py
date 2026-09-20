from types import SimpleNamespace
import unittest

import gymnasium as gym
import torch

from src.ppo_stackcube import AblationAgent, Agent


class DummyEnv:
    def __init__(self):
        self.unwrapped = SimpleNamespace(
            single_action_space=gym.spaces.Box(
                low=-1.0, high=1.0, shape=(9,), dtype=float
            )
        )


def make_args(experiment, frame_stack=1):
    return SimpleNamespace(
        experiment=experiment,
        proprio_dim=25,
        frame_stack=frame_stack,
        obs_norm_clip=5.0,
    )


def make_obs(frame_stack=1, batch_size=2):
    return {
        "rgb": torch.randint(
            0,
            256,
            (batch_size, 64, 64, 6 * frame_stack),
            dtype=torch.uint8,
        ),
        "state": torch.randn((batch_size, 48)),
    }


def parameter_shapes(module):
    return [tuple(parameter.shape) for parameter in module.parameters()]


class B1B4ContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.env = DummyEnv()

    def test_b1_actor_matches_b0_actor_architecture(self):
        obs = make_obs()
        b0_obs = {"rgb": obs["rgb"], "state": obs["state"][..., :25]}
        b0 = Agent(self.env, b0_obs)
        b1 = AblationAgent(self.env, obs, make_args("b1"))

        self.assertEqual(
            parameter_shapes(b0.feature_net),
            parameter_shapes(b1.actor_encoder),
        )
        self.assertEqual(
            parameter_shapes(b0.actor_mean),
            parameter_shapes(b1.actor_mean),
        )
        self.assertEqual(b1.critic_encoder.net[0].in_features, 48)

    def test_actor_never_uses_privileged_dimensions(self):
        for experiment, frame_stack in (
            ("b1", 1),
            ("b2", 1),
            ("b3", 4),
            ("b4", 4),
        ):
            with self.subTest(experiment=experiment):
                obs_a = make_obs(frame_stack)
                obs_b = {
                    "rgb": obs_a["rgb"],
                    "state": obs_a["state"].clone(),
                }
                obs_b["state"][..., 25:] = torch.randn_like(
                    obs_b["state"][..., 25:]
                ) * 1000
                agent = AblationAgent(
                    self.env, obs_a, make_args(experiment, frame_stack)
                )
                with torch.no_grad():
                    action_a = agent.get_action(obs_a, deterministic=True)
                    action_b = agent.get_action(obs_b, deterministic=True)
                torch.testing.assert_close(action_a, action_b)

    def test_b2_changes_only_the_visual_camera_encoder(self):
        obs = make_obs()
        b1 = AblationAgent(self.env, obs, make_args("b1"))
        b2 = AblationAgent(self.env, obs, make_args("b2"))

        self.assertFalse(b1.actor_encoder.separate_cameras)
        self.assertTrue(b2.actor_encoder.separate_cameras)
        self.assertEqual(
            parameter_shapes(b1.actor_encoder.proprio_encoder),
            parameter_shapes(b2.actor_encoder.proprio_encoder),
        )
        self.assertEqual(
            parameter_shapes(b1.actor_mean), parameter_shapes(b2.actor_mean)
        )
        self.assertEqual(
            parameter_shapes(b1.critic_encoder),
            parameter_shapes(b2.critic_encoder),
        )
        self.assertEqual(b1.actor_encoder.image_encoder.net[0][0].in_channels, 6)
        self.assertEqual(b2.actor_encoder.base_encoder.net[0][0].in_channels, 3)
        self.assertEqual(b2.actor_encoder.hand_encoder.net[0][0].in_channels, 3)

    def test_b3_only_expands_each_camera_to_four_frames(self):
        b2 = AblationAgent(self.env, make_obs(), make_args("b2"))
        b3 = AblationAgent(self.env, make_obs(4), make_args("b3", 4))

        self.assertTrue(b3.actor_encoder.separate_cameras)
        self.assertEqual(b3.actor_encoder.frame_stack, 4)
        self.assertEqual(b3.actor_encoder.base_encoder.net[0][0].in_channels, 12)
        self.assertEqual(b3.actor_encoder.hand_encoder.net[0][0].in_channels, 12)
        self.assertEqual(
            parameter_shapes(b2.actor_encoder.proprio_encoder),
            parameter_shapes(b3.actor_encoder.proprio_encoder),
        )
        self.assertEqual(
            parameter_shapes(b2.actor_mean), parameter_shapes(b3.actor_mean)
        )
        self.assertEqual(
            parameter_shapes(b2.critic_encoder),
            parameter_shapes(b3.critic_encoder),
        )

    def test_norm_variants_keep_the_same_trainable_architecture(self):
        for experiment, frame_stack in (
            ("b1", 1),
            ("b2", 1),
            ("b3", 4),
            ("b4", 4),
        ):
            with self.subTest(experiment=experiment):
                obs = make_obs(frame_stack)
                raw_agent = AblationAgent(
                    self.env, obs, make_args(experiment, frame_stack)
                )
                norm_agent = AblationAgent(
                    self.env, obs, make_args(f"{experiment}_norm", frame_stack)
                )
                self.assertEqual(
                    parameter_shapes(raw_agent), parameter_shapes(norm_agent)
                )
                self.assertIsNone(raw_agent.state_normalizer)
                self.assertEqual(tuple(norm_agent.state_normalizer.mean.shape), (48,))

    def test_norm_is_featurewise_and_preserves_actor_information_boundary(self):
        obs_a = make_obs()
        obs_b = {
            "rgb": obs_a["rgb"],
            "state": obs_a["state"].clone(),
        }
        obs_b["state"][..., 25:] += 1000.0
        agent = AblationAgent(self.env, obs_a, make_args("b1_norm"))
        agent.state_normalizer.update(torch.cat([obs_a["state"], obs_b["state"]]))
        normalized_a = agent.normalize_observation(obs_a, update=False)
        normalized_b = agent.normalize_observation(obs_b, update=False)

        self.assertIs(normalized_a["rgb"], obs_a["rgb"])
        self.assertEqual(tuple(normalized_a["state"].shape), (2, 48))
        torch.testing.assert_close(
            normalized_a["state"][..., :25],
            normalized_b["state"][..., :25],
        )
        with torch.no_grad():
            action_a = agent.get_action(normalized_a, deterministic=True)
            action_b = agent.get_action(normalized_b, deterministic=True)
        torch.testing.assert_close(action_a, action_b)

    def test_b4_norm_auxiliary_target_uses_normalized_privileged_slice(self):
        obs = make_obs(4)
        agent = AblationAgent(self.env, obs, make_args("b4_norm", 4))
        normalized = agent.normalize_observation(obs, update=True)
        privilege_target = normalized["state"][..., 25:]
        raw_privilege = obs["state"][..., 25:]
        self.assertEqual(tuple(privilege_target.shape), (2, 23))
        self.assertFalse(torch.equal(privilege_target, raw_privilege))
        with torch.no_grad():
            aux_prediction = agent.get_action_and_value(normalized)[-1]
        self.assertEqual(aux_prediction.shape, privilege_target.shape)

    def test_b4_only_adds_visual_privileged_prediction_head(self):
        obs = make_obs(4)
        b3 = AblationAgent(self.env, obs, make_args("b3", 4))
        b4 = AblationAgent(self.env, obs, make_args("b4", 4))

        self.assertIsNone(b3.aux_head)
        self.assertIsNotNone(b4.aux_head)
        self.assertEqual(b4.aux_head[0].in_features, 256)
        self.assertEqual(b4.aux_head[-1].out_features, 23)
        self.assertEqual(
            parameter_shapes(b3.actor_encoder),
            parameter_shapes(b4.actor_encoder),
        )
        self.assertEqual(
            parameter_shapes(b3.actor_mean), parameter_shapes(b4.actor_mean)
        )
        self.assertEqual(
            parameter_shapes(b3.critic_encoder),
            parameter_shapes(b4.critic_encoder),
        )
        self.assertEqual(
            parameter_shapes(b3.critic), parameter_shapes(b4.critic)
        )

        changed_state = {
            "rgb": obs["rgb"],
            "state": torch.randn_like(obs["state"]) * 1000,
        }
        with torch.no_grad():
            aux_a = b4.get_action_and_value(obs)[-1]
            aux_b = b4.get_action_and_value(changed_state)[-1]
        self.assertEqual(tuple(aux_a.shape), (2, 23))
        torch.testing.assert_close(aux_a, aux_b)


if __name__ == "__main__":
    unittest.main()

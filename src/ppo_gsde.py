# B0-Norm PPO with generalized State-Dependent Exploration (gSDE).
# Paper: https://arxiv.org/abs/2005.05719
# Reference implementation: https://github.com/DLR-RM/stable-baselines3
# Original PPO base: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

@dataclass
class Args:
    # 命令行配置入口：包含环境/观测设置、PPO 超参数和运行控制选项。
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"
    """the environment rendering mode"""

    experiment: str = "b0_norm_gsde"
    """use b0_norm_gsde for paper-style gSDE or b0_norm as its control"""
    proprio_dim: int = 25
    """number of non-privileged robot-state features available to the actor"""
    frame_stack: int = 1
    """number of RGB frames given to the actor; B3/B4 default to four at runtime"""
    aux_coef: float = 0.1
    """B4 privileged-state prediction loss coefficient"""
    obs_norm_clip: float = 5.0
    """absolute clipping bound after B0-Norm proprio standardization"""
    sde_sample_freq: int = 4
    """gSDE exploration-matrix resampling interval; -1 means once per rollout"""
    sde_log_std_init: float = -2.0
    """initial log standard deviation of gSDE exploration weights"""
    sde_epsilon: float = 1e-6
    """numerical floor added to the state-dependent action variance"""
    sde_match_baseline_std: bool = True
    """calibrate initial effective action std to the B0-Norm Gaussian baseline"""
    sde_target_action_std: float = float(np.exp(-0.5))
    """target initial marginal action std when calibration is enabled"""

    # Algorithm specific arguments
    env_id: str = "StackCube-v1"
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    eval_seed: Optional[int] = None
    """evaluation reset seed; intended for final checkpoint evaluation"""
    eval_episodes: int = 100
    """number of completed episodes collected in evaluate-only mode"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 10
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # 正交初始化全连接层，帮助 actor/critic 在训练初期保持较稳定的尺度。
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def should_resample_sde(step, sample_freq):
    """Follow the paper/SB3 convention; -1 means once per rollout."""
    return sample_freq > 0 and step % sample_freq == 0


def action_delta_statistics(action, previous_action, valid, action_range):
    """Return sums used for the paper's normalized continuity cost."""
    if previous_action is None or not torch.any(valid):
        zero = torch.zeros((), device=action.device)
        return zero, zero, 0
    normalized_delta = (
        action[valid] - previous_action[valid]
    ) / action_range
    return (
        normalized_delta.square().sum(),
        normalized_delta.abs().sum(),
        normalized_delta.numel(),
    )


class DictArray(object):
    # 按观测字典结构预分配 rollout 缓冲区，例如分别保存 rgb 和可选的 state。
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            if isinstance(element_space, dict):
                self.data = {}
                for k, v in element_space.items():
                    if v.dtype in (torch.float32, torch.float64):
                        dtype = torch.float32
                    else:
                        dtype = v.dtype
                    self.data[k] = torch.zeros(
                        buffer_shape + tuple(v.shape[1:]), dtype=dtype, device=device
                    )
                return
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


def restrict_policy_observation(obs, experiment: str, proprio_dim: int = 25):
    """Apply the P0/B0 information boundary before data reaches a model/buffer."""
    experiment = experiment.lower()
    if experiment == "p0":
        return {"rgb": obs["rgb"]}
    if experiment in {"b0", "b0_norm", "b0_norm_gsde"}:
        if "state" not in obs:
            raise AssertionError("B0 requires a flattened state observation")
        if obs["state"].shape[-1] != 48:
            raise AssertionError(
                f"B0 expects StackCube's 48-D state, got {obs['state'].shape[-1]}"
            )
        return {
            "rgb": obs["rgb"],
            "state": obs["state"][..., :proprio_dim],
        }
    return obs


class RunningMeanStd(nn.Module):
    """Numerically stable feature-wise running statistics saved in checkpoints."""

    def __init__(self, shape, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.detach().to(torch.float64)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], dtype=torch.float64, device=x.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count)
        self.count.copy_(total_count)

    def normalize(self, x, clip: float):
        mean = self.mean.to(dtype=x.dtype)
        var = self.var.to(dtype=x.dtype)
        normalized = (x - mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(normalized, -clip, clip)


class NatureCNN(nn.Module):
    # 视觉特征提取器：RGB 图像走 CNN；观测中若含 state，则另行编码后拼接。
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        # 图像从 HWC 转为 CNN 所需的 BCHW，并缩放到 [0, 1]；各模态的特征最后拼接。
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    """B0 policy with either diagonal Gaussian exploration or paper-style gSDE."""

    def __init__(
        self,
        envs,
        sample_obs,
        normalize_state=False,
        obs_norm_clip=5.0,
        use_gsde=False,
        sde_log_std_init=-2.0,
        sde_epsilon=1e-6,
    ):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        self.obs_norm_clip = obs_norm_clip
        self.state_normalizer = (
            RunningMeanStd(sample_obs["state"].shape[-1])
            if normalize_state
            else None
        )
        latent_size = self.feature_net.out_features
        action_dim = int(np.prod(envs.unwrapped.single_action_space.shape))
        policy_latent_dim = 512

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        # This is the same 512-unit actor architecture as B0-Norm, split only
        # so the last policy features can drive the gSDE noise function.
        self.actor_latent = nn.Sequential(
            layer_init(nn.Linear(latent_size, policy_latent_dim)),
            nn.ReLU(inplace=True),
        )
        self.actor_mean = layer_init(
            nn.Linear(policy_latent_dim, action_dim),
            std=0.01 * np.sqrt(2),
        )

        self.use_gsde = use_gsde
        self.sde_epsilon = sde_epsilon
        if use_gsde:
            # Full feature-by-action exploration matrix, as in the paper/SB3.
            self.actor_logstd = nn.Parameter(
                torch.full(
                    (policy_latent_dim, action_dim),
                    float(sde_log_std_init),
                )
            )
        else:
            self.actor_logstd = nn.Parameter(
                torch.full((1, action_dim), -0.5)
            )
        self._sde_exploration_matrices = None

    def normalize_observation(self, obs, update=False):
        if self.state_normalizer is None:
            return obs
        state = obs["state"]
        if update:
            self.state_normalizer.update(state)
        normalized_obs = dict(obs)
        normalized_obs["state"] = self.state_normalizer.normalize(
            state, self.obs_norm_clip
        )
        return normalized_obs

    def get_features(self, obs):
        return self.feature_net(obs)

    def get_value(self, obs):
        features = self.feature_net(obs)
        return self.critic(features)

    def _policy_distribution(self, obs):
        features = self.feature_net(obs)
        policy_latent = self.actor_latent(features)
        action_mean = self.actor_mean(policy_latent)
        if self.use_gsde:
            # The paper uses policy features for epsilon(s; theta_epsilon).
            # Detaching matches the PPO gSDE implementation in SB3:
            # distribution gradients update log_std but not the features.
            latent_sde = policy_latent.detach()
            weight_std = torch.exp(self.actor_logstd)
            variance = latent_sde.square() @ weight_std.square()
            action_std = torch.sqrt(variance + self.sde_epsilon)
        else:
            action_std = torch.exp(
                self.actor_logstd.expand_as(action_mean)
            )
        return features, policy_latent, action_mean, Normal(
            action_mean, action_std
        )

    def reset_noise(self, num_envs):
        """Sample one exploration matrix per parallel environment."""
        if not self.use_gsde:
            return
        weight_std = torch.exp(self.actor_logstd).detach()
        base_noise = torch.randn(
            (num_envs,) + tuple(weight_std.shape),
            device=weight_std.device,
            dtype=weight_std.dtype,
        )
        self._sde_exploration_matrices = (
            base_noise * weight_std.unsqueeze(0)
        )

    def _sample_gsde_action(self, action_mean, policy_latent):
        batch_size = policy_latent.shape[0]
        if (
            self._sde_exploration_matrices is None
            or self._sde_exploration_matrices.shape[0] != batch_size
        ):
            self.reset_noise(batch_size)
        latent_sde = policy_latent.detach().unsqueeze(1)
        exploration = torch.bmm(
            latent_sde, self._sde_exploration_matrices
        ).squeeze(1)
        return action_mean + exploration

    def get_action_std(self, obs):
        return self._policy_distribution(obs)[-1].scale

    @torch.no_grad()
    def calibrate_sde_action_std(self, obs, target_action_std):
        """Match mean initial marginal std without changing the gSDE equation."""
        if not self.use_gsde:
            raise RuntimeError("gSDE calibration requires use_gsde=True")
        features = self.feature_net(obs)
        policy_latent = self.actor_latent(features)
        latent_norm = torch.sqrt(
            policy_latent.detach().square().sum(dim=1)
            + self.sde_epsilon
        )
        mean_latent_norm = latent_norm.mean()
        target = torch.as_tensor(
            target_action_std,
            dtype=self.actor_logstd.dtype,
            device=self.actor_logstd.device,
        )
        calibrated_log_std = torch.log(target / mean_latent_norm)
        self.actor_logstd.fill_(calibrated_log_std)
        self._sde_exploration_matrices = None
        return calibrated_log_std.item()

    def get_action(self, obs, deterministic=False):
        _, policy_latent, action_mean, distribution = (
            self._policy_distribution(obs)
        )
        if deterministic:
            return action_mean
        if self.use_gsde:
            return self._sample_gsde_action(action_mean, policy_latent)
        return distribution.sample()

    def get_action_and_value(self, obs, action=None):
        # PPO uses the analytical state-dependent Gaussian marginal for
        # log-probability ratios, while rollout actions use a temporally
        # repeated exploration matrix.
        features, policy_latent, action_mean, distribution = (
            self._policy_distribution(obs)
        )
        if action is None:
            if self.use_gsde:
                action = self._sample_gsde_action(
                    action_mean, policy_latent
                )
            else:
                action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(1),
            distribution.entropy().sum(1),
            self.critic(features),
            None,
        )

class FrameStacker:
    """GPU-side RGB history that correctly resets individual vector environments."""

    def __init__(self, num_stack: int):
        self.num_stack = num_stack
        self.history = None
        self.channels_per_frame = None

    def reset(self, obs):
        self.channels_per_frame = obs["rgb"].shape[-1]
        self.history = torch.cat([obs["rgb"]] * self.num_stack, dim=-1)
        out = dict(obs)
        out["rgb"] = self.history
        return out

    def step(self, obs, done):
        assert self.history is not None
        current = obs["rgb"]
        if self.num_stack == 1:
            self.history = current
        else:
            self.history = torch.cat(
                [self.history[..., self.channels_per_frame :], current], dim=-1
            )
            reset_history = torch.cat([current] * self.num_stack, dim=-1)
            mask = done.bool().reshape((-1,) + (1,) * (self.history.ndim - 1))
            self.history = torch.where(mask, reset_history, self.history)
        out = dict(obs)
        out["rgb"] = self.history
        return out


class ImageEncoder(nn.Module):
    def __init__(self, sample_rgb, out_features):
        super().__init__()
        in_channels = sample_rgb.shape[-1]
        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = cnn(
                sample_rgb.float().permute(0, 3, 1, 2).cpu()
            ).shape[1]
        self.net = nn.Sequential(
            cnn,
            nn.Linear(n_flatten, out_features),
            nn.ReLU(),
        )

    def forward(self, rgb):
        return self.net(rgb.float().permute(0, 3, 1, 2) / 255.0)


class VisionProprioEncoder(nn.Module):
    """Actor observation encoder; privileged object state is never consumed here."""

    def __init__(self, sample_obs, proprio_dim, frame_stack, separate_cameras):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.frame_stack = frame_stack
        self.separate_cameras = separate_cameras
        total_channels = sample_obs["rgb"].shape[-1]
        assert total_channels % frame_stack == 0
        self.channels_per_frame = total_channels // frame_stack
        assert self.channels_per_frame == 6, (
            "StackCube experiments expect base_camera RGB plus hand_camera RGB "
            f"(6 channels per frame), got {self.channels_per_frame}"
        )
        if separate_cameras:
            base, hand = self._split_cameras(sample_obs["rgb"])
            self.base_encoder = ImageEncoder(base, 128)
            self.hand_encoder = ImageEncoder(hand, 128)
        else:
            self.image_encoder = ImageEncoder(sample_obs["rgb"], 256)
        self.proprio_encoder = nn.Linear(proprio_dim, 256)
        self.out_features = 512

    def _split_cameras(self, rgb):
        b, h, w, _ = rgb.shape
        frames = rgb.reshape(b, h, w, self.frame_stack, self.channels_per_frame)
        base = frames[..., :3].reshape(b, h, w, self.frame_stack * 3)
        hand = frames[..., 3:6].reshape(b, h, w, self.frame_stack * 3)
        return base, hand

    def encode(self, obs):
        if self.separate_cameras:
            base, hand = self._split_cameras(obs["rgb"])
            visual = torch.cat(
                [self.base_encoder(base), self.hand_encoder(hand)], dim=-1
            )
        else:
            visual = self.image_encoder(obs["rgb"])
        proprio = self.proprio_encoder(obs["state"][..., : self.proprio_dim])
        return torch.cat([visual, proprio], dim=-1), visual

    def forward(self, obs):
        actor_features, _ = self.encode(obs)
        return actor_features


class PrivilegedStateEncoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU())
        self.out_features = 512

    def forward(self, obs):
        return self.net(obs["state"])


class AblationAgent(nn.Module):
    """Unified network used by the B1-B4/R1 ablation chain."""

    def __init__(self, envs, sample_obs, args):
        super().__init__()
        experiment = args.experiment.lower()
        experiment = {
            "a1": "b1",
            "a2": "b2",
            "a3": "b3",
            "a4": "b4",
        }.get(experiment, experiment)
        assert experiment in {"v0_sep", "b1", "b2", "b3", "b4", "r1"}
        self.experiment = experiment
        separate_cameras = experiment in {"b2", "b3", "b4", "r1"}
        self.proprio_dim = args.proprio_dim
        self.actor_encoder = VisionProprioEncoder(
            sample_obs,
            proprio_dim=args.proprio_dim,
            frame_stack=args.frame_stack,
            separate_cameras=separate_cameras,
        )
        if experiment == "v0_sep":
            self.critic_encoder = VisionProprioEncoder(
                sample_obs,
                proprio_dim=args.proprio_dim,
                frame_stack=args.frame_stack,
                separate_cameras=False,
            )
        else:
            self.critic_encoder = PrivilegedStateEncoder(
                sample_obs["state"].shape[-1]
            )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5
        )
        privilege_dim = sample_obs["state"].shape[-1] - args.proprio_dim
        self.aux_head = (
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, privilege_dim))
            if experiment == "b4"
            else None
        )

    def get_value(self, obs):
        return self.critic(self.critic_encoder(obs))

    def get_action(self, obs, deterministic=False):
        actor_features, _ = self.actor_encoder.encode(obs)
        action_mean = self.actor_mean(actor_features)
        if deterministic:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        return Normal(action_mean, action_std).sample()

    def get_action_and_value(self, obs, action=None):
        actor_features, visual_features = self.actor_encoder.encode(obs)
        action_mean = self.actor_mean(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        value = self.critic(self.critic_encoder(obs))
        aux_prediction = (
            self.aux_head(visual_features) if self.aux_head is not None else None
        )
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
            aux_prediction,
        )

class Logger:
    # 将训练和评估的标量写入 TensorBoard；开启 track 时也同步到 Weights & Biases。
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    # 1. 解析参数，并由并行环境数和每轮采样步数推导 batch 大小及总更新轮数。
    args = tyro.cli(Args)
    args.experiment = args.experiment.lower()
    valid_experiments = {
        "p0", "b0", "b0_norm", "b0_norm_gsde", "b1", "b2", "b3", "b4", "r1",
        "v0_sep", "a1", "a2", "a3", "a4",
    }
    if args.experiment not in valid_experiments:
        raise ValueError(
            f"unknown experiment {args.experiment!r}; choose from {sorted(valid_experiments)}"
        )
    if args.experiment in {"p0", "b0", "b0_norm", "b0_norm_gsde"} and args.proprio_dim != 25:
        raise ValueError("P0/B0/B0-Norm contract requires --proprio-dim 25")
    if args.experiment == "p0":
        args.include_state = False
    elif args.experiment in {"b0", "b0_norm", "b0_norm_gsde"}:
        args.include_state = True
    temporal_experiments = {"b3", "b4", "a3", "a4"}
    if args.experiment in temporal_experiments and args.frame_stack == 1:
        args.frame_stack = 4
    if args.experiment not in temporal_experiments and args.frame_stack != 1:
        raise ValueError("frame_stack > 1 is reserved for B3/B4")
    if args.experiment == "r1":
        args.partial_reset = False
    if args.proprio_dim <= 0:
        raise ValueError("proprio_dim must be positive")
    if args.experiment == "b0_norm_gsde":
        if args.sde_sample_freq == 0 or args.sde_sample_freq < -1:
            raise ValueError("--sde-sample-freq must be -1 or a positive integer")
        if args.sde_epsilon <= 0:
            raise ValueError("--sde-epsilon must be positive")
        if args.sde_target_action_std <= 0:
            raise ValueError("--sde-target-action-std must be positive")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 2. 创建 ManiSkill 训练/评估环境：请求 RGB 观测，并使用 PhysX CUDA 仿真后端。
    env_kwargs = dict(obs_mode="rgb+state", render_mode=args.render_mode, sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # 整理观测字典：保留 RGB、不使用深度图；include_state 决定是否额外拼入状态向量。
    # include_state=False 才是纯 RGB；True 时 Actor 和 Critic 都会看到 RGB 与 state 的融合特征。
    include_flat_state = args.experiment != "p0"
    envs = FlattenRGBDObservationWrapper(
        envs, rgb=True, depth=False, state=include_flat_state
    )
    eval_envs = FlattenRGBDObservationWrapper(
        eval_envs, rgb=True, depth=False, state=include_flat_state
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    # 将多个并行 episode 封装为向量环境，集中进行采样并记录 episode 指标。
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Initialize raw observations, then add per-environment RGB history.
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    initial_eval_seed = (
        args.eval_seed if args.evaluate and args.eval_seed is not None else args.seed
    )
    eval_obs, _ = eval_envs.reset(seed=initial_eval_seed)
    next_obs = restrict_policy_observation(
        next_obs, args.experiment, args.proprio_dim
    )
    eval_obs = restrict_policy_observation(
        eval_obs, args.experiment, args.proprio_dim
    )
    train_stacker = FrameStacker(args.frame_stack)
    eval_stacker = FrameStacker(args.frame_stack)
    next_obs = train_stacker.reset(next_obs)
    eval_obs = eval_stacker.reset(eval_obs)
    next_done = torch.zeros(
        args.num_envs if not args.evaluate else 1, device=device
    )
    print("####")
    print(
        f"experiment={args.experiment} frame_stack={args.frame_stack} "
        f"partial_reset={args.partial_reset} target_kl={args.target_kl} "
        f"sde_sample_freq={args.sde_sample_freq} "
        f"sde_log_std_init={args.sde_log_std_init} "
        f"sde_match_baseline_std={args.sde_match_baseline_std}"
    )
    print(
        f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} "
        f"args.num_eval_envs={args.num_eval_envs}"
    )
    print(
        f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} "
        f"args.update_epochs={args.update_epochs}"
    )
    observation_shapes = " ".join(
        f"{key}={tuple(value.shape)}" for key, value in next_obs.items()
    )
    print(f"observation {observation_shapes}")
    print("####")
    if args.experiment in {"p0", "b0", "b0_norm", "b0_norm_gsde"}:
        agent = Agent(
            envs,
            sample_obs=next_obs,
            normalize_state=args.experiment in {"b0_norm", "b0_norm_gsde"},
            obs_norm_clip=args.obs_norm_clip,
            use_gsde=args.experiment == "b0_norm_gsde",
            sde_log_std_init=args.sde_log_std_init,
            sde_epsilon=args.sde_epsilon,
        ).to(device)
    else:
        agent = AblationAgent(envs, sample_obs=next_obs, args=args).to(device)
    print(f"model_parameters={sum(p.numel() for p in agent.parameters())}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.experiment in {"b0_norm", "b0_norm_gsde"}:
        next_obs = agent.normalize_observation(next_obs, update=not args.evaluate)
        eval_obs = agent.normalize_observation(eval_obs, update=False)

    if (
        args.experiment == "b0_norm_gsde"
        and args.checkpoint is None
        and args.sde_match_baseline_std
    ):
        calibrated_log_std = agent.calibrate_sde_action_std(
            next_obs, args.sde_target_action_std
        )
        print(
            "calibrated_sde_log_std="
            f"{calibrated_log_std:.6f} "
            f"target_action_std={args.sde_target_action_std:.6f}"
        )
        if logger is not None:
            logger.add_scalar(
                "charts/sde_initial_log_std", calibrated_log_std, 0
            )

    # Rollout buffers are unnecessary in evaluate-only mode.
    if not args.evaluate:
        obs = DictArray(
            (args.num_steps, args.num_envs), next_obs, device=device
        )
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape,
            device=device,
        )
        logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        dones = torch.zeros((args.num_steps, args.num_envs), device=device)
        values = torch.zeros((args.num_steps, args.num_envs), device=device)

    cumulative_times = defaultdict(float)
    previous_action = None
    action_range = torch.as_tensor(
        envs.single_action_space.high - envs.single_action_space.low,
        dtype=torch.float32,
        device=device,
    )

    for iteration in range(1, args.num_iterations + 1):
        # 一次迭代包含：定期评估、采集一批并行轨迹、计算优势，最后做 PPO 参数更新。
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        continuity_square_sum = torch.zeros((), device=device)
        continuity_absolute_sum = torch.zeros((), device=device)
        continuity_count = 0
        agent.eval()
        if getattr(agent, "use_gsde", False):
            agent.reset_noise(args.num_envs)
        if iteration % args.eval_freq == 1:
            # 评估使用确定性动作（Actor 均值），统计 episode 指标，并按配置录制视频。
            print("Evaluating")
            stime = time.perf_counter()
            evaluation_seed = (
                args.eval_seed if args.evaluate and args.eval_seed is not None else None
            )
            eval_raw_obs, _ = eval_envs.reset(seed=evaluation_seed)
            eval_raw_obs = restrict_policy_observation(
                eval_raw_obs, args.experiment, args.proprio_dim
            )
            if args.experiment in {"b0_norm", "b0_norm_gsde"}:
                eval_raw_obs = agent.normalize_observation(
                    eval_raw_obs, update=False
                )
            eval_obs = eval_stacker.reset(eval_raw_obs)
            eval_metrics = defaultdict(list)
            num_episodes = 0
            eval_steps = 0
            eval_previous_action = None
            eval_previous_done = torch.zeros(
                args.num_eval_envs, dtype=torch.bool, device=device
            )
            eval_continuity_square_sum = torch.zeros((), device=device)
            eval_continuity_absolute_sum = torch.zeros((), device=device)
            eval_continuity_count = 0
            while True:
                with torch.no_grad():
                    eval_action = agent.get_action(
                        eval_obs, deterministic=True
                    )
                    eval_valid_delta = (
                        ~eval_previous_done
                        if eval_previous_action is not None
                        else torch.zeros(
                            args.num_eval_envs,
                            dtype=torch.bool,
                            device=device,
                        )
                    )
                    (
                        eval_square_sum,
                        eval_absolute_sum,
                        eval_delta_count,
                    ) = action_delta_statistics(
                        eval_action,
                        eval_previous_action,
                        eval_valid_delta,
                        action_range,
                    )
                    eval_continuity_square_sum += eval_square_sum
                    eval_continuity_absolute_sum += eval_absolute_sum
                    eval_continuity_count += eval_delta_count
                    eval_previous_action = eval_action.detach().clone()
                    eval_raw_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    eval_raw_obs = restrict_policy_observation(
                        eval_raw_obs, args.experiment, args.proprio_dim
                    )
                    if args.experiment in {"b0_norm", "b0_norm_gsde"}:
                        eval_raw_obs = agent.normalize_observation(
                            eval_raw_obs, update=False
                        )
                    eval_done = torch.logical_or(eval_terminations, eval_truncations)
                    eval_previous_done = eval_done.bool()
                    eval_obs = eval_stacker.step(eval_raw_obs, eval_done)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        completed = int(mask.sum().item())
                        if args.evaluate:
                            completed = min(
                                completed, args.eval_episodes - num_episodes
                            )
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v[mask][:completed])
                        num_episodes += completed
                eval_steps += 1
                if args.evaluate:
                    if num_episodes >= args.eval_episodes:
                        break
                elif eval_steps >= args.num_eval_steps:
                    break
            print(
                f"Evaluated {eval_steps * args.num_eval_envs} environment steps "
                f"resulting in {num_episodes} episodes"
            )
            for k, v in eval_metrics.items():
                mean = torch.cat(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if eval_continuity_count > 0:
                eval_continuity_cost = (
                    100.0
                    * eval_continuity_square_sum.item()
                    / eval_continuity_count
                )
                eval_mean_abs_delta = (
                    eval_continuity_absolute_sum.item()
                    / eval_continuity_count
                )
                print(
                    "eval_continuity_cost_mean="
                    f"{eval_continuity_cost}"
                )
                print(
                    "eval_mean_abs_normalized_action_delta="
                    f"{eval_mean_abs_delta}"
                )
                if logger is not None:
                    logger.add_scalar(
                        "eval/continuity_cost",
                        eval_continuity_cost,
                        global_step,
                    )
                    logger.add_scalar(
                        "eval/mean_abs_normalized_action_delta",
                        eval_mean_abs_delta,
                        global_step,
                    )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            # 4. 保存当前观测和旧策略统计量，再在所有并行环境中执行采样到的动作。
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            if (
                getattr(agent, "use_gsde", False)
                and should_resample_sde(step, args.sde_sample_freq)
            ):
                agent.reset_noise(args.num_envs)
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            valid_delta = (
                torch.ones(args.num_envs, dtype=torch.bool, device=device)
                if previous_action is not None
                else torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            )
            valid_delta &= ~next_done.bool()
            square_sum, absolute_sum, delta_count = action_delta_statistics(
                action, previous_action, valid_delta, action_range
            )
            continuity_square_sum += square_sum
            continuity_absolute_sum += absolute_sum
            continuity_count += delta_count
            previous_action = action.detach().clone()

            # TRY NOT TO MODIFY: execute the game and log data.
            raw_next_obs, reward, terminations, truncations, infos = envs.step(action)
            raw_next_obs = restrict_policy_observation(
                raw_next_obs, args.experiment, args.proprio_dim
            )
            if args.experiment in {"b0_norm", "b0_norm_gsde"}:
                raw_next_obs = agent.normalize_observation(
                    raw_next_obs, update=True
                )
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            next_obs = train_stacker.step(raw_next_obs, next_done)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                final_observation = restrict_policy_observation(
                    infos["final_observation"],
                    args.experiment,
                    args.proprio_dim,
                )
                if args.experiment in {"b0_norm", "b0_norm_gsde"}:
                    final_observation = agent.normalize_observation(
                        final_observation, update=False
                    )
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_observation).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # 5. 用 Critic bootstrap 并计算 GAE 优势与 returns；final_values 处理 episode 结束时的截断。
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # 将 [时间步, 环境编号] 两维摊平，形成 PPO 更新用的样本批次。
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 6. PPO 更新：打乱 rollout 样本，按 minibatch 重复多轮优化 Actor 和 Critic。
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        aux_loss = torch.zeros((), device=device)
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, aux_prediction = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Clipped surrogate 策略损失：限制新旧策略概率比，避免单次策略更新过大。
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic 回归 GAE returns；可选 value clipping，限制价值估计的变化幅度。
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                if aux_prediction is not None:
                    privilege_target = b_obs["state"][
                        mb_inds, args.proprio_dim :
                    ]
                    aux_loss = nn.functional.mse_loss(
                        aux_prediction, privilege_target
                    )
                # 总损失由策略损失、熵正则项和加权价值损失组成。
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + args.aux_coef * aux_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/aux_loss", aux_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        if continuity_count > 0:
            logger.add_scalar(
                "charts/continuity_cost_train",
                100.0 * continuity_square_sum.item() / continuity_count,
                global_step,
            )
            logger.add_scalar(
                "charts/mean_abs_normalized_action_delta",
                continuity_absolute_sum.item() / continuity_count,
                global_step,
            )
        if hasattr(agent, "get_action_std"):
            with torch.no_grad():
                action_std_mean = agent.get_action_std(next_obs).mean()
            logger.add_scalar(
                "charts/action_std_mean", action_std_mean.item(), global_step
            )
        if getattr(agent, "use_gsde", False):
            logger.add_scalar(
                "charts/sde_weight_std_mean",
                torch.exp(agent.actor_logstd).mean().item(),
                global_step,
            )
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    # 训练结束保存最终权重；评估模式不写训练用的 final checkpoint。
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()

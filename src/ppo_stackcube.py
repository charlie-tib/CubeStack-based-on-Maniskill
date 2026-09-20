# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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
from torch.distributions.beta import Beta
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

    experiment: str = "p0"
    """one of: p0, b0, b1, b2, b3, b4, r1 (a1-a4 are legacy aliases)"""
    proprio_dim: int = 25
    """number of non-privileged robot-state features available to the actor"""
    frame_stack: int = 1
    """number of RGB frames given to the actor; B3/B4 default to four at runtime"""
    aux_coef: float = 0.1
    """B4 privileged-state prediction loss coefficient"""
    obs_norm_clip: float = 5.0
    """absolute clipping bound after B0-Norm proprio standardization"""
    policy_distribution: str = "gaussian"
    """actor distribution: gaussian (legacy default) or beta"""
    caps_temporal_coef: float = 0.0
    """temporal CAPS loss coefficient; zero preserves the baseline"""
    caps_pre_success_only: bool = False
    """apply temporal CAPS only until the first success in each episode"""
    drac_policy_coef: float = 0.0
    """Actor-only DrAC policy consistency coefficient"""
    rgb_photometric_aug: bool = False
    """enable shared mild RGB brightness/contrast augmentation for DrAC"""

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
    if experiment in {"b0", "b0_norm"}:
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
    # Actor 和 Critic 共享特征网络；Actor 输出连续动作分布，Critic 估计状态价值 V(s)。
    def __init__(self, envs, sample_obs, normalize_state=False, obs_norm_clip=5.0):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        self.obs_norm_clip = obs_norm_clip
        self.state_normalizer = (
            RunningMeanStd(sample_obs["state"].shape[-1])
            if normalize_state
            else None
        )
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
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
    def get_features(self, x):
        return self.feature_net(x)
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        # 评估时使用动作均值；训练时从高斯分布采样，以保留探索。
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        # PPO 更新所需接口：返回动作、log-prob、熵和价值；传入旧动作时评估其新策略概率。
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
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


def shared_rgb_photometric_augmentation(
    obs, brightness_limit=0.05, contrast_limit=0.10
):
    """Apply one brightness/contrast sample to every camera/frame of a sample."""
    rgb = obs["rgb"]
    batch_shape = (rgb.shape[0],) + (1,) * (rgb.ndim - 1)
    contrast = 1.0 + (
        2.0 * torch.rand(batch_shape, device=rgb.device) - 1.0
    ) * contrast_limit
    brightness = (
        2.0 * torch.rand(batch_shape, device=rgb.device) - 1.0
    ) * brightness_limit * 255.0
    augmented = (rgb.float() - 127.5) * contrast + 127.5 + brightness
    augmented = augmented.clamp_(0.0, 255.0)
    if not rgb.dtype.is_floating_point:
        augmented = augmented.round().to(rgb.dtype)
    else:
        augmented = augmented.to(rgb.dtype)
    augmented_obs = dict(obs)
    augmented_obs["rgb"] = augmented
    return augmented_obs


def gaussian_policy_kl(mean_p, logstd_p, mean_q, logstd_q):
    """Mean KL[p || q] for diagonal Gaussians, summed over action dims."""
    logstd_delta = logstd_p - logstd_q
    mean_delta = mean_p - mean_q
    per_dimension = (
        -logstd_delta
        + 0.5 * torch.expm1(2.0 * logstd_delta)
        + 0.5 * mean_delta.square() * torch.exp(-2.0 * logstd_q)
    )
    return per_dimension.sum(dim=-1).mean()


def temporal_caps_loss(action_mean, next_action_mean, valid_mask, action_range):
    """Temporal CAPS loss over valid, same-episode observation pairs only."""
    valid_mask = valid_mask.bool()
    if not torch.any(valid_mask):
        return action_mean.sum() * 0.0
    normalized_delta = (
        next_action_mean[valid_mask] - action_mean[valid_mask]
    ) / action_range
    return normalized_delta.square().mean()


def pre_success_caps_transition_mask(success_seen, step_success, next_done):
    """Mask CAPS from the first successful transition until episode reset."""
    next_done = next_done.bool()
    success_seen = success_seen.bool() | step_success.bool()
    valid = ~(success_seen | next_done)
    success_seen = torch.where(next_done, False, success_seen)
    return valid, success_seen


def rollout_action_diagnostics(actions, dones, action_low, action_high):
    """Diagnostics used to catch action-bound and continuity pathologies."""
    action_range = action_high - action_low
    outside = torch.logical_or(actions < action_low, actions > action_high)
    margin = 0.01 * action_range
    saturated = torch.logical_or(
        actions <= action_low + margin, actions >= action_high - margin
    )
    finite = torch.isfinite(actions)
    diagnostics = {
        "action_batch_std": actions.float().std(unbiased=False),
        "action_abs_max": actions.float().abs().max(),
        "action_out_of_bounds_fraction": outside.float().mean(),
        "action_saturation_fraction": saturated.float().mean(),
        "action_nonfinite_fraction": (~finite).float().mean(),
    }
    valid = ~dones[1:].bool()
    if actions.shape[0] > 1 and torch.any(valid):
        raw_delta = actions[1:] - actions[:-1]
        valid_delta = raw_delta[valid]
        diagnostics["action_delta"] = valid_delta.abs().mean()
        diagnostics["continuity_cost"] = (
            valid_delta / action_range
        ).square().mean()
    else:
        zero = actions.sum() * 0.0
        diagnostics["action_delta"] = zero
        diagnostics["continuity_cost"] = zero
    return diagnostics


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

    def __init__(
        self,
        sample_obs,
        proprio_dim,
        frame_stack,
        separate_cameras,
        state_key="state",
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.frame_stack = frame_stack
        self.separate_cameras = separate_cameras
        self.state_key = state_key
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
        proprio = self.proprio_encoder(
            obs[self.state_key][..., : self.proprio_dim]
        )
        return torch.cat([visual, proprio], dim=-1), visual

    def forward(self, obs):
        actor_features, _ = self.encode(obs)
        return actor_features


class PrivilegedStateEncoder(nn.Module):
    def __init__(self, state_dim, state_key="state"):
        super().__init__()
        self.state_key = state_key
        self.net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU())
        self.out_features = 512

    def forward(self, obs):
        return self.net(obs[self.state_key])


class AblationAgent(nn.Module):
    """Unified network used by the B1-B4/R1 ablation chain."""

    def __init__(self, envs, sample_obs, args):
        super().__init__()
        requested_experiment = args.experiment.lower()
        if requested_experiment == "b1_actor_norm":
            experiment = "b1"
            normalization_scope = "actor"
        elif requested_experiment == "b1_critic_norm":
            experiment = "b1"
            normalization_scope = "critic"
        else:
            normalize_state = requested_experiment.endswith("_norm")
            experiment = (
                requested_experiment[: -len("_norm")]
                if normalize_state
                else requested_experiment
            )
            normalization_scope = "both" if normalize_state else "none"
        experiment = {
            "a1": "b1",
            "a2": "b2",
            "a3": "b3",
            "a4": "b4",
        }.get(experiment, experiment)
        assert experiment in {"v0_sep", "b1", "b2", "b3", "b4", "r1"}
        self.experiment = experiment
        self.normalization_scope = normalization_scope
        self.policy_distribution = getattr(args, "policy_distribution", "gaussian")
        if self.policy_distribution not in {"gaussian", "beta"}:
            raise ValueError(f"unsupported policy distribution: {self.policy_distribution}")
        self.obs_norm_clip = args.obs_norm_clip
        normalizer_dim = (
            args.proprio_dim
            if normalization_scope == "actor"
            else sample_obs["state"].shape[-1]
        )
        self.state_normalizer = (
            RunningMeanStd(normalizer_dim)
            if normalization_scope != "none"
            else None
        )
        separate_cameras = experiment in {"b2", "b3", "b4", "r1"}
        self.proprio_dim = args.proprio_dim
        self.actor_encoder = VisionProprioEncoder(
            sample_obs,
            proprio_dim=args.proprio_dim,
            frame_stack=args.frame_stack,
            separate_cameras=separate_cameras,
            state_key=(
                "actor_state" if normalization_scope == "actor" else "state"
            ),
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
                sample_obs["state"].shape[-1],
                state_key=(
                    "critic_state"
                    if normalization_scope == "critic"
                    else "state"
                ),
            )

        action_space = envs.unwrapped.single_action_space
        action_dim = np.prod(action_space.shape)
        if self.policy_distribution == "gaussian":
            # Keep the legacy module construction order and names checkpoint-compatible.
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(inplace=True),
                layer_init(
                    nn.Linear(512, action_dim),
                    std=0.01 * np.sqrt(2),
                ),
            )
        else:
            self.actor_body = nn.Sequential(
                layer_init(nn.Linear(512, 512)),
                nn.ReLU(inplace=True),
            )
            beta_raw_init = float(np.log(np.expm1(0.01)))
            self.actor_alpha = layer_init(
                nn.Linear(512, action_dim),
                std=0.01 * np.sqrt(2),
                bias_const=beta_raw_init,
            )
            self.actor_beta = layer_init(
                nn.Linear(512, action_dim),
                std=0.01 * np.sqrt(2),
                bias_const=beta_raw_init,
            )
            self.register_buffer(
                "action_low", torch.as_tensor(action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.as_tensor(action_space.high, dtype=torch.float32)
            )
            self.register_buffer("action_scale", self.action_high - self.action_low)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        if self.policy_distribution == "gaussian":
            self.actor_logstd = nn.Parameter(
                torch.ones(1, action_dim) * -0.5
            )
        privilege_dim = sample_obs["state"].shape[-1] - args.proprio_dim
        self.aux_head = (
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, privilege_dim))
            if experiment == "b4"
            else None
        )

    def normalize_observation(self, obs, update=False):
        if self.state_normalizer is None:
            return obs
        state = obs["state"]
        state_to_normalize = (
            state[..., : self.proprio_dim]
            if self.normalization_scope == "actor"
            else state
        )
        if update:
            self.state_normalizer.update(state_to_normalize)
        normalized_obs = dict(obs)
        normalized_state = self.state_normalizer.normalize(
            state_to_normalize, self.obs_norm_clip
        )
        if self.normalization_scope == "actor":
            normalized_obs["actor_state"] = normalized_state
        elif self.normalization_scope == "critic":
            normalized_obs["critic_state"] = normalized_state
        else:
            normalized_obs["state"] = normalized_state
        return normalized_obs

    def get_value(self, obs):
        return self.critic(self.critic_encoder(obs))

    def _beta_distribution_from_features(self, actor_features):
        hidden = self.actor_body(actor_features)
        alpha = torch.nn.functional.softplus(self.actor_alpha(hidden)) + 1.0
        beta = torch.nn.functional.softplus(self.actor_beta(hidden)) + 1.0
        return Beta(alpha, beta)

    def get_action_mean(self, obs):
        actor_features, _ = self.actor_encoder.encode(obs)
        if self.policy_distribution == "gaussian":
            return self.actor_mean(actor_features)
        probs = self._beta_distribution_from_features(actor_features)
        return self.action_low + self.action_scale * probs.mean

    def get_gaussian_policy(self, obs):
        if self.policy_distribution != "gaussian":
            raise RuntimeError("Gaussian policy parameters requested from Beta actor")
        actor_features, _ = self.actor_encoder.encode(obs)
        mean = self.actor_mean(actor_features)
        logstd = self.actor_logstd.expand_as(mean)
        return mean, logstd

    def get_action(self, obs, deterministic=False):
        actor_features, _ = self.actor_encoder.encode(obs)
        if self.policy_distribution == "gaussian":
            action_mean = self.actor_mean(actor_features)
            if deterministic:
                return action_mean
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            return Normal(action_mean, action_std).sample()
        probs = self._beta_distribution_from_features(actor_features)
        if deterministic:
            unit_action = probs.mean
        else:
            unit_action = probs.sample()
        return self.action_low + self.action_scale * unit_action

    def get_action_and_value(self, obs, action=None):
        actor_features, visual_features = self.actor_encoder.encode(obs)
        if self.policy_distribution == "gaussian":
            action_mean = self.actor_mean(actor_features)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            probs = Normal(action_mean, torch.exp(action_logstd))
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        else:
            probs = self._beta_distribution_from_features(actor_features)
            if action is None:
                unit_action = probs.sample()
                action = self.action_low + self.action_scale * unit_action
            else:
                unit_action = (action - self.action_low) / self.action_scale
                unit_action = unit_action.clamp(
                    torch.finfo(unit_action.dtype).eps,
                    1.0 - torch.finfo(unit_action.dtype).eps,
                )
            logprob = (
                probs.log_prob(unit_action) - torch.log(self.action_scale)
            ).sum(1)
            entropy = (
                probs.entropy() + torch.log(self.action_scale)
            ).sum(1)
        value = self.critic(self.critic_encoder(obs))
        aux_prediction = (
            self.aux_head(visual_features) if self.aux_head is not None else None
        )
        return (
            action,
            logprob,
            entropy,
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
        "p0", "b0", "b0_norm", "b1", "b1_norm", "b2", "b2_norm",
        "b3", "b3_norm", "b4", "b4_norm", "r1",
        "b1_actor_norm", "b1_critic_norm",
        "v0_sep", "a1", "a2", "a3", "a4",
    }
    if args.experiment not in valid_experiments:
        raise ValueError(
            f"unknown experiment {args.experiment!r}; choose from {sorted(valid_experiments)}"
        )
    args.policy_distribution = args.policy_distribution.lower()
    if args.policy_distribution not in {"gaussian", "beta"}:
        raise ValueError("--policy-distribution must be gaussian or beta")
    if args.caps_temporal_coef < 0 or args.drac_policy_coef < 0:
        raise ValueError("CAPS and DrAC coefficients must be non-negative")
    if args.caps_pre_success_only and args.caps_temporal_coef == 0:
        raise ValueError("pre-success-only CAPS requires a nonzero CAPS coefficient")
    if args.caps_pre_success_only and args.partial_reset:
        raise ValueError("pre-success-only CAPS requires --no-partial-reset")
    optimization_ablation = (
        args.policy_distribution != "gaussian"
        or args.caps_temporal_coef > 0
        or args.drac_policy_coef > 0
        or args.rgb_photometric_aug
    )
    if optimization_ablation and args.experiment != "b4_norm":
        raise ValueError("Beta/CAPS/DrAC ablations require --experiment b4_norm")
    if args.drac_policy_coef > 0 and not args.rgb_photometric_aug:
        raise ValueError("DrAC requires --rgb-photometric-aug")
    if args.rgb_photometric_aug and args.drac_policy_coef == 0:
        raise ValueError("RGB augmentation is reserved for a nonzero DrAC loss")
    if args.drac_policy_coef > 0 and args.policy_distribution != "gaussian":
        raise ValueError("this Actor-DrAC experiment is defined for Gaussian policy")
    if args.caps_temporal_coef > 0 and args.drac_policy_coef > 0:
        raise ValueError("first-round ablations must not combine CAPS and DrAC")
    if args.experiment in {"p0", "b0", "b0_norm"} and args.proprio_dim != 25:
        raise ValueError("P0/B0/B0-Norm contract requires --proprio-dim 25")
    if args.experiment == "p0":
        args.include_state = False
    elif args.experiment in {"b0", "b0_norm"}:
        args.include_state = True
    temporal_experiments = {"b3", "b3_norm", "b4", "b4_norm", "a3", "a4"}
    if args.experiment in temporal_experiments and args.frame_stack == 1:
        args.frame_stack = 4
    if args.experiment not in temporal_experiments and args.frame_stack != 1:
        raise ValueError("frame_stack > 1 is reserved for B3/B4")
    if args.experiment == "r1":
        args.partial_reset = False
    if args.proprio_dim <= 0:
        raise ValueError("proprio_dim must be positive")
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
    action_low = torch.as_tensor(
        envs.single_action_space.low, dtype=torch.float32, device=device
    )
    action_high = torch.as_tensor(
        envs.single_action_space.high, dtype=torch.float32, device=device
    )
    action_range = action_high - action_low

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
        f"partial_reset={args.partial_reset} target_kl={args.target_kl}"
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
    if args.experiment in {"p0", "b0", "b0_norm"}:
        agent = Agent(
            envs,
            sample_obs=next_obs,
            normalize_state=args.experiment.endswith("_norm"),
            obs_norm_clip=args.obs_norm_clip,
        ).to(device)
    else:
        agent = AblationAgent(envs, sample_obs=next_obs, args=args).to(device)
    print(f"model_parameters={sum(p.numel() for p in agent.parameters())}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.experiment.endswith("_norm"):
        next_obs = agent.normalize_observation(next_obs, update=not args.evaluate)
        eval_obs = agent.normalize_observation(eval_obs, update=False)

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

        caps_next_obs = None
        caps_valid = None
        caps_success_seen = None
        if args.caps_temporal_coef > 0:
            caps_next_obs = DictArray(
                (args.num_steps, args.num_envs), next_obs, device=device
            )
            caps_valid = torch.zeros(
                (args.num_steps, args.num_envs), dtype=torch.bool, device=device
            )
            if args.caps_pre_success_only:
                caps_success_seen = torch.zeros(
                    args.num_envs, dtype=torch.bool, device=device
                )
    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        # 一次迭代包含：定期评估、采集一批并行轨迹、计算优势，最后做 PPO 参数更新。
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
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
            if args.experiment.endswith("_norm"):
                eval_raw_obs = agent.normalize_observation(
                    eval_raw_obs, update=False
                )
            eval_obs = eval_stacker.reset(eval_raw_obs)
            eval_metrics = defaultdict(list)
            num_episodes = 0
            eval_steps = 0
            while True:
                with torch.no_grad():
                    eval_raw_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    eval_raw_obs = restrict_policy_observation(
                        eval_raw_obs, args.experiment, args.proprio_dim
                    )
                    if args.experiment.endswith("_norm"):
                        eval_raw_obs = agent.normalize_observation(
                            eval_raw_obs, update=False
                        )
                    eval_done = torch.logical_or(eval_terminations, eval_truncations)
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
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            raw_next_obs, reward, terminations, truncations, infos = envs.step(action)
            raw_next_obs = restrict_policy_observation(
                raw_next_obs, args.experiment, args.proprio_dim
            )
            if args.experiment.endswith("_norm"):
                raw_next_obs = agent.normalize_observation(
                    raw_next_obs, update=True
                )
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            next_obs = train_stacker.step(raw_next_obs, next_done)
            rewards[step] = reward.view(-1) * args.reward_scale
            if caps_next_obs is not None:
                caps_next_obs[step] = next_obs
                caps_valid[step] = ~next_done.bool()
                if caps_success_seen is not None:
                    step_success = infos.get("success")
                    if step_success is None and "final_info" in infos:
                        step_success = infos["final_info"].get("success")
                    if step_success is None:
                        raise KeyError(
                            "pre-success-only CAPS requires info['success']"
                        )
                    pre_success_valid, caps_success_seen = (
                        pre_success_caps_transition_mask(
                            caps_success_seen,
                            step_success.to(device=device),
                            next_done,
                        )
                    )
                    caps_valid[step] &= pre_success_valid

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
                if args.experiment.endswith("_norm"):
                    final_observation = agent.normalize_observation(
                        final_observation, update=False
                    )
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_observation).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        action_diagnostics = rollout_action_diagnostics(
            actions, dones, action_low, action_high
        )
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
        if caps_next_obs is not None:
            b_caps_next_obs = caps_next_obs.reshape((-1,))
            b_caps_valid = caps_valid.reshape(-1)

        # 6. PPO 更新：打乱 rollout 样本，按 minibatch 重复多轮优化 Actor 和 Critic。
        agent.train()
        caps_loss = torch.zeros((), device=device)
        drac_loss = torch.zeros((), device=device)
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
                caps_loss = torch.zeros((), device=device)
                drac_loss = torch.zeros((), device=device)
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
                if args.caps_temporal_coef > 0:
                    caps_action_mean = agent.get_action_mean(b_obs[mb_inds])
                    caps_next_action_mean = agent.get_action_mean(
                        b_caps_next_obs[mb_inds]
                    )
                    caps_loss = temporal_caps_loss(
                        caps_action_mean,
                        caps_next_action_mean,
                        b_caps_valid[mb_inds],
                        action_range,
                    )
                if args.drac_policy_coef > 0:
                    original_mean, original_logstd = agent.get_gaussian_policy(
                        b_obs[mb_inds]
                    )
                    augmented_obs = shared_rgb_photometric_augmentation(b_obs[mb_inds])
                    augmented_mean, augmented_logstd = agent.get_gaussian_policy(
                        augmented_obs
                    )
                    drac_loss = gaussian_policy_kl(
                        original_mean.detach(), original_logstd.detach(),
                        augmented_mean, augmented_logstd,
                    )
                # 总损失由 PPO、B4 辅助目标和单个可选正则项组成。
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + args.aux_coef * aux_loss
                    + args.caps_temporal_coef * caps_loss
                    + args.drac_policy_coef * drac_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite PPO/B4/regularization loss detected"
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
        logger.add_scalar("losses/caps_temporal_loss", caps_loss.item(), global_step)
        logger.add_scalar("losses/drac_policy_kl", drac_loss.item(), global_step)
        logger.add_scalar("losses/aux_loss", aux_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        for name, value in action_diagnostics.items():
            logger.add_scalar(f"diagnostics/{name}", value.item(), global_step)
        if caps_valid is not None:
            logger.add_scalar(
                "diagnostics/caps_valid_fraction",
                caps_valid.float().mean().item(),
                global_step,
            )
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
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

import os
from threading import Lock

import numpy as np
import torch
from torch.autograd import Variable

from ray.rllib.dqn.dqn_evaluator import adjust_nstep
from ray.rllib.optimizers.policy_evaluator import PolicyEvaluator
from ray.rllib.optimizers.sample_batch import SampleBatch
from ray.rllib.utils.atari_wrappers import wrap_deepmind
from ray.rllib.utils.compression import pack

from agent import Agent
from env import Env
from main import parse_args
from test import from_gym


class RainbowEvaluator(PolicyEvaluator):
    """Integration point with RLlib policy optimizers.
    
    This wraps the existing torch agent to implement the right interface."""

    def __init__(self, config, env_creator):
        self.config = config
        self.config.update(config)
        self.args = parse_args([
            "--multi-step={}".format(self.config["n_step"]),
            "--discount={}".format(self.config["gamma"]),
            "--lr={}".format(self.config["lr"]),
        ])
        self.env = wrap_deepmind(
            env_creator(self.config["env_config"]), dim=84)
        self.action_space = self.env.action_space.n
        self.torch_agent = Agent(self.args, self.action_space)
        self.state = from_gym(self.env.reset())
        self.local_timestep = 0
        self.episode_rewards = [0.0]
        self.episode_lengths = [0.0]

    def sample(self):
        obs, actions, rewards, new_obs, dones = [], [], [], [], []
        for _ in range(
                self.config["sample_batch_size"] + self.config["n_step"] - 1):
            action = self.torch_agent.act(self.state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = from_gym(next_state)
            obs.append(self.state.data.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            new_obs.append(next_state.data.cpu().numpy())
            dones.append(1.0 if done else 0.0)
            self.state = next_state
            self.episode_rewards[-1] += reward
            self.episode_lengths[-1] += 1
            if done:
                self.state = from_gym(self.env.reset())
                self.torch_agent.reset_noise()
                self.episode_rewards.append(0.0)
                self.episode_lengths.append(0.0)
            self.local_timestep += 1

        # N-step Q adjustments
        if self.config["n_step"] > 1:
            # Adjust for steps lost from truncation
            self.local_timestep -= (self.config["n_step"] - 1)
            adjust_nstep(
                self.config["n_step"], self.config["gamma"],
                obs, actions, rewards, new_obs, dones)

        batch = SampleBatch({
            "obs": obs, "actions": actions, "rewards": rewards,
            "new_obs": new_obs, "dones": dones,
            "weights": np.ones_like(rewards)})
        assert batch.count == self.config["sample_batch_size"]

        td_errors = self.torch_agent.compute_td_error(batch)
        batch.data["obs"] = [pack(o) for o in batch["obs"]]
        batch.data["new_obs"] = [pack(o) for o in batch["new_obs"]]
        new_priorities = (
            np.abs(td_errors) + self.config["prioritized_replay_eps"])
        batch.data["weights"] = new_priorities

        return batch

    def compute_gradients(self, samples):
        grad, td_error = self.torch_agent.grad(samples)
        return grad, {"td_error": td_error}

    def apply_gradients(self, grads):
        return self.torch_agent.apply_grad(grads)

    def compute_apply(self, samples):
        td_error = self.torch_agent.compute_apply(samples)
        return {"td_error": td_error}

    def get_weights(self):
        out = {}
        for k, v in self.torch_agent.policy_net.state_dict().items():
            out[k] = v.cpu()
        return out

    def set_weights(self, weights):
        self.torch_agent.policy_net.load_state_dict(weights)
        self.torch_agent.target_net.load_state_dict(weights)

    def stats(self):
        mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 5)
        mean_100ep_length = round(np.mean(self.episode_lengths[-101:-1]), 5)
        return {
            "mean_100ep_reward": mean_100ep_reward,
            "mean_100ep_length": mean_100ep_length,
            "num_episodes": len(self.episode_rewards),
            "local_timestep": self.local_timestep,
        }

    def update_target(self):
        self.torch_agent.update_target_net()

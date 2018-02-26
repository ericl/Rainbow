import os
from threading import Lock

import numpy as np
import torch
from torch.autograd import Variable

from ray.rllib.dqn.dqn_evaluator import adjust_nstep
from ray.rllib.dqn.common.atari_wrappers import wrap_deepmind
from ray.rllib.optimizers.evaluator import Evaluator
from ray.rllib.optimizers.sample_batch import SampleBatch, pack

from agent import Agent
from env import Env
from main import parse_args


def to_rainbow(obs):
    tensor = torch.from_numpy(obs.transpose((2,0,1)))
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


class RainbowEvaluator(Evaluator):
    
    def __init__(self, config, env_creator):
        self.config = config
        self.config.update(config)
        self.args = parse_args([
            "--multi-step={}".format(self.config["n_step"]),
            "--discount={}".format(self.config["gamma"]),
            "--lr={}".format(self.config["lr"]),
            "--game={}".format(self.config["env"]),
        ])
        self.env = wrap_deepmind(
            env_creator(self.config["env_config"]),
            frame_stack=True, scale=True)
        self.action_space = self.env.action_space.n
        self.agent = Agent(self.args, self.action_space)
        self.state = to_rainbow(self.env.reset())
        self.local_timestep = 0
        self.episode_rewards = [0.0]
        self.episode_lengths = [0.0]
        self.lock = Lock()

    def sample(self):
        obs, actions, rewards, new_obs, dones = [], [], [], [], []
        for _ in range(
                self.config["sample_batch_size"] + self.config["n_step"] - 1):
            action = self.agent.act(self.state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = to_rainbow(next_state)
            obs.append(self.state.data.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            new_obs.append(next_state.data.cpu().numpy())
            dones.append(1.0 if done else 0.0)
            self.state = next_state
            self.episode_rewards[-1] += reward
            self.episode_lengths[-1] += 1
            if done:
                self.state = to_rainbow(self.env.reset())
                self.agent.reset_noise()
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

        td_errors = self.agent.compute_td_error(batch)
        batch.data["obs"] = [pack(o) for o in batch["obs"]]
        batch.data["new_obs"] = [pack(o) for o in batch["new_obs"]]
        new_priorities = (
            np.abs(td_errors) + self.config["prioritized_replay_eps"])
        batch.data["weights"] = new_priorities

        return batch

    def compute_gradients(self, samples):
        return self.agent.grad(samples)

    def apply_gradients(self, grads):
        return self.agent.apply_grad(grads)

    def compute_apply(self, samples):
        with self.lock:
            return self.agent.compute_apply(samples)

    def get_weights(self):
        with self.lock:
            out = {}
            for k, v in self.agent.policy_net.state_dict().items():
                out[k] = v.cpu()
            return out

    def set_weights(self, weights):
        self.agent.policy_net.load_state_dict(weights)
        self.agent.target_net.load_state_dict(weights)

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
        self.agent.update_target_net()

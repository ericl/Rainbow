from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import numpy as np
from torch.autograd import Variable

import ray
from ray.rllib.agent import Agent
from ray.rllib.optimizers.local_sync_replay import LocalSyncReplayOptimizer
from ray.rllib.optimizers.apex_optimizer import ApexOptimizer
from ray.rllib.utils.actors import split_colocated
from ray.tune.result import TrainingResult

from rainbow_evaluator import RainbowEvaluator


DEFAULT_CONFIG = dict(
    # N-step Q learning
    n_step=3,
    # Discount factor for the MDP
    gamma=0.99,
    # Arguments to pass to the env creator
    env_config={},

    # === Exploration ===
    # Number of env steps to optimize for before returning
    timesteps_per_iteration=1000,
    # How many steps of the model to sample before learning starts.
    learning_starts=1000,
    # Update the target network every `target_network_update_freq` steps.
    target_network_update_freq=500,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then each
    # worker will have a replay buffer of this size.
    buffer_size=50000,
    # If True prioritized replay buffer will be used.
    prioritized_replay=True,
    # Alpha parameter for prioritized replay buffer.
    prioritized_replay_alpha=0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    prioritized_replay_beta=0.4,
    # Epsilon to add to the TD errors when updating priorities.
    prioritized_replay_eps=1e-6,

    # === Optimization ===
    # Learning rate for adam optimizer
    lr=5e-4,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    sample_batch_size=4,
    # Size of a batched sampled from replay buffer for training. Note that if
    # async_updates is set, then each worker returns gradients for a batch of
    # this size.
    train_batch_size=32,

    # === Parallelism ===
    num_workers=0,
    # Max number of steps to delay synchronizing weights of workers.
    max_weight_sync_delay=400,
    # Number of Apex replay shards.
    num_replay_buffer_shards=1,
    # Whether to use the Apex optimizer.
    apex=False)


class RainbowRLlibAgent(Agent):
    """Agent for integration with Ray tune etc.

    This is mostly copy-pasted from the DQNAgent in RLlib."""

    _agent_name = "RainbowApex"
    _allow_unknown_subkeys = ["optimizer", "env_config"]
    _default_config = DEFAULT_CONFIG

    def _init(self):
        if self.config["apex"]:
            self.optimizer = ApexOptimizer.make(
                RainbowEvaluator, [self.config, self.env_creator],
                self.config["num_workers"], {
                    k: self.config[k]
                    for k in [
                        "learning_starts", "buffer_size", "sample_batch_size",
                        "prioritized_replay", "prioritized_replay_alpha",
                        "prioritized_replay_beta", "prioritized_replay_eps",
                        "train_batch_size", "sample_batch_size",
                        "num_replay_buffer_shards", "max_weight_sync_delay"
                    ]
                })
        else:
            self.optimizer = LocalSyncReplayOptimizer.make(
                RainbowEvaluator, [self.config, self.env_creator], 0, {
                    k: self.config[k]
                    for k in [
                        "learning_starts", "buffer_size", "sample_batch_size",
                        "prioritized_replay", "prioritized_replay_alpha",
                        "prioritized_replay_beta", "prioritized_replay_eps",
                        "train_batch_size", "sample_batch_size"
                    ]
                })

        self.last_target_update_ts = 0
        self.num_target_updates = 0

    @property
    def global_timestep(self):
        return self.optimizer.num_steps_sampled

    def update_target_if_needed(self):
        if self.global_timestep - self.last_target_update_ts > \
                self.config["target_network_update_freq"]:
            self.optimizer.local_evaluator.update_target()
            self.last_target_update_ts = self.global_timestep
            self.num_target_updates += 1

    def _train(self):
        start_timestep = self.global_timestep

        while (self.global_timestep - start_timestep <
               self.config["timesteps_per_iteration"]):
            self.optimizer.step()
            self.update_target_if_needed()

        return self._train_stats(start_timestep)

    def _train_stats(self, start_timestep):
        if self.optimizer.remote_evaluators:
            stats = ray.get([
                e.stats.remote() for e in self.optimizer.remote_evaluators])
        else:
            stats = self.optimizer.local_evaluator.stats()
            if not isinstance(stats, list):
                stats = [stats]

        mean_100ep_reward = 0.0
        mean_100ep_length = 0.0
        num_episodes = 0

        for s in stats:
            mean_100ep_reward += s["mean_100ep_reward"] / len(stats)
            mean_100ep_length += s["mean_100ep_length"] / len(stats)
            num_episodes += s["num_episodes"]

        opt_stats = self.optimizer.stats()

        result = TrainingResult(
            episode_reward_mean=mean_100ep_reward,
            episode_len_mean=mean_100ep_length,
            episodes_total=num_episodes,
            timesteps_this_iter=self.global_timestep - start_timestep,
            info=dict({
                "num_target_updates": self.num_target_updates,
            }, **opt_stats))

        return result

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.optimizer.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        raise NotImplementedError

    def _restore(self, checkpoint_path):
        raise NotImplementedError

    def compute_action(self, observation):
        raise NotImplementedError

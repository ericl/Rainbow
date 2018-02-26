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
    # Number of workers for collecting samples with. Note that the typical
    # setting is 1 unless your environment is particularly slow to sample.
    num_workers=1,
    # Max number of steps to delay synchronizing weights of workers.
    max_weight_sync_delay=400,
    num_replay_buffer_shards=1,
    force_remote_evaluators=False,
    apex=False)


class RainbowRLlibAgent(Agent):
    _agent_name = "RainbowApex"
    _allow_unknown_subkeys = ["optimizer", "env_config"]
    _default_config = DEFAULT_CONFIG

    def _init(self):
        self.local_evaluator = RainbowEvaluator(self.config, self.env_creator)
        remote_cls = ray.remote(num_cpus=1)(RainbowEvaluator)
        self.remote_evaluators = [
            remote_cls.remote(self.config, self.env_creator)
            for i in range(self.config["num_workers"])]
        if self.config["force_remote_evaluators"]:
            _, self.remote_evaluators = split_colocated(
                self.remote_evaluators)
        if self.config["apex"]:
            self.optimizer = ApexOptimizer(
                self.config, self.local_evaluator, self.remote_evaluators)
        else:
            self.optimizer = LocalSyncReplayOptimizer(
                self.config, self.local_evaluator, self.remote_evaluators)

        self.global_timestep = 0
        self.last_target_update_ts = 0
        self.num_target_updates = 0

    def _train(self):
        start_timestep = self.global_timestep
        num_steps = 0

        while (self.global_timestep - start_timestep <
               self.config["timesteps_per_iteration"]):

            self.global_timestep += self.optimizer.step()
            num_steps += 1

            if self.global_timestep - self.last_target_update_ts > \
                    self.config["target_network_update_freq"]:
                self.local_evaluator.update_target()
                self.last_target_update_ts = self.global_timestep
                self.num_target_updates += 1

        test_stats = self._update_global_stats()
        mean_100ep_reward = 0.0
        mean_100ep_length = 0.0
        num_episodes = 0
        explorations = []

        for s in test_stats:
            mean_100ep_reward += s["mean_100ep_reward"] / len(test_stats)
            mean_100ep_length += s["mean_100ep_length"] / len(test_stats)
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

    def _update_global_stats(self):
        if self.remote_evaluators:
            stats = ray.get([
                e.stats.remote() for e in self.remote_evaluators])
        else:
            stats = self.local_evaluator.stats()
            if not isinstance(stats, list):
                stats = [stats]
        new_timestep = sum(s["local_timestep"] for s in stats)
        assert new_timestep >= self.global_timestep, new_timestep
        self.global_timestep = new_timestep

        return stats

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        raise NotImplementedError

    def _restore(self, checkpoint_path):
        raise NotImplementedError

    def compute_action(self, observation):
        return self.local_evaluator.agent.act(Variable(observation))

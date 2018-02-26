#!/usr/bin/env python

import multiprocessing
import sys

import ray
from ray.tune import register_trainable, run_experiments

from rainbow_rllib_agent import RainbowRLlibAgent

register_trainable("Rainbow", RainbowRLlibAgent)

ray.init(num_gpus=1)

run_experiments({
    "rainbow-simple-pong": {
        "run": "Rainbow",
        "env": "PongNoFrameskip-v4",
        "resources": {
            "cpu": 1,
            "gpu": 1,
        },
        "config": {
            "num_workers": 0,
            "apex": False,
            "lr": .0001,
            "n_step": 3,
            "gamma": 0.99,
            "sample_batch_size": 4,
            "train_batch_size": 32,
            "force_remote_evaluators": False,
            "learning_starts": 80000,
            "buffer_size": 1000000,
            "target_network_update_freq": 32000,
            "timesteps_per_iteration": 1000,
        },
    },
})

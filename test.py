# -*- coding: utf-8 -*-

import os
import gym
import plotly
from plotly.graph_objs import Scatter, Line
import torch
from torch.autograd import Variable

from env import Env
from ray.rllib.dqn.common.atari_wrappers import wrap_deepmind


# Globals
Ts, rewards, Qs = [], [], []

def to_rainbow(obs, volatile=False):
    tensor = torch.from_numpy(obs.transpose((2,0,1)))
    if torch.cuda.is_available():
        return Variable(tensor.cuda(), volatile)
    else:
        return Variable(tensor, volatile)

# Test DQN
def test(args, T, dqn, val_mem, evaluate=False):
  env = wrap_deepmind(
      gym.make(args.game), frame_stack=True, scale=True)
  Ts.append(T)
  T_rewards, T_Qs = [], []

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = to_rainbow(env.reset(), volatile=True), 0, False

      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done, _ = env.step(action)  # Step
      state = to_rainbow(state, volatile=True)
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  # Return average reward and Q-value
  return sum(T_rewards) / len(T_rewards)


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = torch.Tensor(ys_population)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)

import random
from collections import namedtuple
import torch
from torch.autograd import Variable


# Segment tree data structure where parent node values are sum of children node values
class SumTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.data = []  # Wrap-around cyclic buffer

  def append(self, data):
    if len(self.data) < self.index + 1:
      self.data.append(None)
    self.data[self.index] = data  # Store data in underlying data structure
    self.index = (self.index + 1) % self.size  # Update index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal', 'priority'))


class ReplayMemory():
  def __init__(self, args, capacity):
    self.dtype_byte = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor
    self.dtype_long = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    self.dtype_float = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
    self.t = 0  # Internal episode timestep counter
    self.transitions = SumTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Add empty states to prepare for new episode
  def preappend(self):
    self.t = 0
    # Blank transitions from before episode
    for h in range(-self.history + 1, 0):
      # Add blank state with zero priority
      self.transitions.append(Transition(h, torch.ByteTensor(84, 84).zero_(), None, 0, True, 0))

  # Adds state, action and reward at time t (technically reward from time t + 1, but kept at t for all buffers to be in sync)
  def append(self, state, action, reward):
    state = state[-1].mul(255).byte().cpu()  # Only store last frame and discretise to save memory
    # Store new transition with maximum priority
    self.transitions.append(Transition(self.t, state, action, reward, True, 1))
    self.t += 1

  # Add empty state at end of episode
  def postappend(self):
    # Add blank transitions (used to replace terminal state) with zero priority; simplifies truncated n-step discounted return calculations
    for _ in range(self.n):
      self.transitions.append(Transition(self.t, torch.ByteTensor(84, 84).zero_(), None, 0, False, 0))

  def sample(self, batch_size):
    idxs = self.dtype_long(random.sample(range(len(self.transitions.data)), batch_size))
    # If any transitions straddle current index, remove them (simpler than replacing with unique valid transitions) TODO: Separate out pre- and post-index
    valid_idxs = idxs.sub(self.transitions.index).abs_() > max(self.history, self.n)
    # If any transitions have 0 probability (priority), remove them (may not be necessary check)
    probs = self.dtype_float([self.transitions.get(i).priority for i in idxs])
    valid_idxs.mul_(probs != 0)
    idxs = idxs[valid_idxs]

    # Retrieve all required transition data (from t - h to t + n)
    full_transitions = [[self.transitions.get(i + t) for i in idxs] for t in range(1 - self.history, self.n + 1)]  # Time x batch

    # Create stack of states and nth next states
    state_stack, next_state_stack = [], []
    for t in range(self.history):
      state_stack.append(torch.stack([transition.state for transition in full_transitions[t]], 0))
      next_state_stack.append(torch.stack([transition.state for transition in full_transitions[t + self.n]], 0))  # nth next state
    states = Variable(torch.stack(state_stack, 1).type(self.dtype_float).div_(255))  # Un-discretise
    next_states = Variable(torch.stack(next_state_stack, 1).type(self.dtype_float).div_(255), volatile=True)

    actions = self.dtype_long([transition.action for transition in full_transitions[self.history - 1]])

    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
    returns = [transition.reward for transition in full_transitions[self.history - 1]]
    for n in range(1, self.n):
      # Invalid nth next states have reward 0 and hence do not affect calculation
      returns = [R + self.discount ** n * transition.reward for R, transition in zip(returns, full_transitions[self.history + n - 1])]
    returns = self.dtype_float(returns)

    nonterminals = self.dtype_float([transition.nonterminal for transition in full_transitions[self.history + self.n - 1]]).unsqueeze(1)  # Mask for non-terminal nth next states

    return states, actions, returns, next_states, nonterminals

  # Set up internal state for iterator
  def __iter__(self):
    # Find indices for valid samples
    self.valid_idxs = []
    for t in range(self.capacity):
      if self.transitions.data[t].timestep >= 0:
        self.valid_idxs.append(t)
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == len(self.valid_idxs):
      raise StopIteration
    # Create stack of states and nth next states
    state_stack = []
    for t in reversed(range(self.history)):
      state_stack.append(self.transitions.data[self.valid_idxs[self.current_idx - t]].state)
    state = Variable(torch.stack(state_stack, 0).type(self.dtype_float).div_(255), volatile=True)  # Agent will turn into batch
    self.current_idx += 1
    return state

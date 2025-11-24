import random
import numpy as np
from collections import deque

class OffPolicyReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen = buffer_size)

    def remember(self, state, action, reward, done, next_state):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
    

class OnPolicyReplayBuffer:
    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_advantages_and_returns(self, last_value, done):
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae_lam = 0
        
        # We need to iterate backwards
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                # This is the last step in the trajectory
                next_non_terminal = 1.0 - done
                next_value = last_value

            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
                
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + np.array(self.values, dtype = np.float32)
        return advantages, returns

    def get_batch(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.values),
        )
    
    def __len__(self):
        return len(self.states)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from replay_buffers import OffPolicyReplayBuffer

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, state, reparameterize=True):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if reparameterize:
            # Reparameterization trick
            # x = mean + std * N(0,1)
            x_t = dist.rsample()

        else:
            x_t = dist.sample()
        
        # Squash action and calculate log_prob
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Calculate log_prob
        # This is the correction factor for the tanh squashing
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim = -1, keepdim = True)
        
        return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 1)

        # Q2 architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, 128)
        self.layer_5 = nn.Linear(128, 128)
        self.layer_6 = nn.Linear(128, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)

        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)
        return q1, q2

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr, critic_lr,
                 buffer_size, batch_size, gamma = 0.99, tau = 0.005, alpha = 0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        
        # Automatic temperature (alpha) tuning
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad = True, device = self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = actor_lr)
        self.alpha = alpha

        self.buffer = OffPolicyReplayBuffer(buffer_size)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action

    def soft_update_target_networks(self):
        for target_param, main_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
    
    @torch.no_grad()
    def get_action(self, state, exploration_noise = 0.0): # SAC explores via entropy
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.actor.eval()
        action, _ = self.actor.sample(state_tensor, reparameterize = False)
        self.actor.train()

        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def get_deterministic_action(self, state):
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0).to(self.device)

        self.actor.eval()
        # For SAC eval, we take the mean of the distribution
        mean, _ = self.actor(state_tensor)
        action = torch.tanh(mean) * self.max_action
        self.actor.train()

        return action.squeeze(0).cpu().numpy()

    def remember(self, state, action, reward, done, next_state):
        self.buffer.remember(state, action, reward, done, next_state)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None # actor, critic, alpha losses
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype = torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype = torch.float32).unsqueeze(1).to(self.device)

        # --- Critic Loss ---
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            
            # Bellman equation with entropy
            td_target = rewards + (1 - dones) * self.gamma * (q_target - self.alpha * next_log_prob)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Loss ---
        pi, log_pi = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_pi - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Alpha (Temperature) Loss ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item() # Update alpha value
        
        # --- Update Target Networks ---
        self.soft_update_target_networks()
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()
        
    def save_models(self, save_dir):
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))

    def load_models(self, load_dir):
        self.actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pth"), map_location = self.device))
        self.critic.load_state_dict(torch.load(os.path.join(load_dir, "critic.pth"), map_location = self.device))
        
        self.critic_target.load_state_dict(self.critic.state_dict())
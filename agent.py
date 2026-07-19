import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dueling_dqn import DuelingDQN

class D3QNAgent:
    """
    D3QN (Dueling Double Deep Q-Network) Agent implementation.
    Decouples action selection from evaluation (Double DQN) and
    splits the Q-value estimation into state-value and advantage components (Dueling).
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4, 
                 gamma: float = 0.99, buffer_size: int = 100000, batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Dual Networks
        self.online_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        
        # Hard sync target network at start
        self.update_target_network()
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        
        # Exploration Parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy.
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
            return torch.argmax(q_values).item()
            
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                         next_state: np.ndarray, done: bool):
        """
        Saves environment transitions to replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def train_step(self) -> float:
        """
        Performs one gradient descent step of D3QN updates.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample random experience mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert list to numpy array first to avoid torch warning about list performance
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current predicted Q-Values: Q(s, a; online_net_weights)
        current_q = self.online_net(states).gather(1, actions)
        
        # DOUBLE DQN STEP:
        # 1. Action selection via Online Network: a* = argmax_a Q(s_next, a; online_net)
        # 2. Evaluation via Target Network: Q(s_next, a*; target_net)
        with torch.no_grad():
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_values
            
        # Optimize Mean Squared Error Loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
        
    def update_target_network(self):
        """
        Hard synchronization copy of network weights from online to target network.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target_network(self, tau: float = 0.005):
        """
        Soft synchronization of target network: theta_target = tau * theta_online + (1 - tau) * theta_target.
        """
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

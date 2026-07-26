import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dueling_dqn import VLAGuardedDuelingDQN

class D3QNAgent:
    """
    VLA-Guarded D3QN (Dueling Double Deep Q-Network) Agent implementation.
    
    Decouples action selection from evaluation (Double DQN) and conditions
    Q-value estimations on high-level VLA semantic modulation vectors.
    """
    def __init__(self, state_dim: int = 28, action_dim: int = 5, semantic_dim: int = 64,
                 lr: float = 1e-4, gamma: float = 0.99, buffer_size: int = 100000, batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.semantic_dim = semantic_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Dual Networks with VLA Semantic Conditioning
        self.online_net = VLAGuardedDuelingDQN(state_dim, action_dim, semantic_dim).to(self.device)
        self.target_net = VLAGuardedDuelingDQN(state_dim, action_dim, semantic_dim).to(self.device)
        
        # Hard sync target network at start
        self.update_target_network()
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        
        # Latency & Asynchronous Guarding: Cache latest VLA semantic embedding vector
        self.cached_semantic_emb = np.zeros(semantic_dim, dtype=np.float32)
        
        # Exploration Parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
    def update_vla_guard_embedding(self, semantic_vec: np.ndarray):
        """
        Asynchronously updates the cached VLA semantic modulation embedding vector.
        Called whenever the high-level VLM (e.g. 1-2 Hz) emits a new behavior token.
        """
        self.cached_semantic_emb = np.array(semantic_vec, dtype=np.float32)
        
    def select_action(self, state: np.ndarray, semantic_emb: np.ndarray = None, evaluate: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy conditioned on VLA semantic embedding.
        Executes instantly at 50 Hz using cached semantic_emb.
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        if semantic_emb is None:
            semantic_emb = self.cached_semantic_emb
            
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        sem_t = torch.FloatTensor(semantic_emb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.online_net(state_t, sem_t)
            return torch.argmax(q_values).item()
            
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                         next_state: np.ndarray, done: bool, semantic_emb: np.ndarray = None):
        """
        Saves environment transitions including VLA semantic embedding to replay memory.
        """
        if semantic_emb is None:
            semantic_emb = self.cached_semantic_emb
        self.memory.append((state, action, reward, next_state, done, semantic_emb))
        
    def train_step(self) -> float:
        """
        Performs one gradient descent step of VLA-Guarded D3QN updates.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample random experience mini-batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Check transition tuple format for backward compatibility
        if len(batch[0]) == 6:
            states, actions, rewards, next_states, dones, sem_embs = zip(*batch)
            sem_embs = torch.FloatTensor(np.array(sem_embs)).to(self.device)
        else:
            states, actions, rewards, next_states, dones = zip(*batch)
            sem_embs = None
            
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current predicted Q-Values: Q(s, a; e_vla)
        current_q = self.online_net(states, sem_embs).gather(1, actions)
        
        # DOUBLE DQN STEP WITH VLA CONDITIONING:
        with torch.no_grad():
            best_next_actions = self.online_net(next_states, sem_embs).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states, sem_embs).gather(1, best_next_actions)
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

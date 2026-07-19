import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    """
    Dueling DQN Network architecture as specified in the reference implementation.
    Splits the state representation into a state-value stream V(s) and 
    an advantage stream A(s, a).
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingDQN, self).__init__()
        
        # Shared Feature Extraction Layers
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # State Value Stream Head - V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Action Advantage Stream Head - A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_network(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Aggregate streams using the mean advantage normalization formula:
        # Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
        q_vals = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_vals

import torch
import torch.nn as nn

class VLAGuardedDuelingDQN(nn.Module):
    """
    VLA-Guarded Dueling Deep Q-Network (VLA-D3QN).
    
    Solves the 'Semantic-Safety Gap in Discrete Deep Reinforcement Learning'
    by dynamically conditioning low-level action-values (Q-values) on high-level 
    VLA (Vision-Language-Action) semantic modulation embeddings (e.g. from Prismatic / LLaVA-Phi).
    
    Architecture Hierarchy:
    1. Sensor Encoder: Maps 28-dim LiDAR + Kinematic state to 128-dim latent space.
    2. VLA Semantic Encoder: Maps dense semantic modulation vector (e.g. 64-dim) to 128-dim latent space.
    3. Multi-Modal Fusion Layer: Fuses sensory and semantic features into a 256-dim joint latent vector.
    4. Conditioned Dueling Streams:
       - Value Stream V(s, e_vla): Shifts baseline expectation based on room semantics.
       - Advantage Stream A(s, a, e_vla): Alters relative action benefits (e.g. penalizing thrust in crowded zones).
    """
    def __init__(self, state_dim: int = 28, action_dim: int = 5, semantic_dim: int = 64):
        super(VLAGuardedDuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.semantic_dim = semantic_dim
        
        # 1. Raw Sensor Feature Encoder (24 LiDAR + 4 Kinematics)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 2. VLA Semantic Vector Projection Head
        self.vla_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 3. Multi-Modal Feature Fusion MLP
        # Fuses 128-dim sensor features + 128-dim VLA semantic features -> 256-dim joint latent vector
        self.fusion_network = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # 4. Semantically Conditioned State-Value Stream Head - V(s, e_vla)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 5. Semantically Conditioned Action Advantage Stream Head - A(s, a, e_vla)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state: torch.Tensor, semantic_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass supporting asynchronous VLA semantic embedding caching.
        
        Args:
            state: Raw sensor state tensor of shape [batch_size, state_dim]
            semantic_emb: Cached VLA semantic vector of shape [batch_size, semantic_dim].
                          If None, defaults to zero vector for 50Hz zero-latency fallback execution.
        """
        # Handle asynchronous execution caching: if semantic_emb is not provided, use cached/zero fallback
        if semantic_emb is None:
            semantic_emb = torch.zeros((state.shape[0], self.semantic_dim), device=state.device)
            
        # Sensor Encoding (LiDAR + Goal Features)
        sensor_feat = self.sensor_encoder(state)
        
        # VLA Semantic Modulation Encoding
        vla_feat = self.vla_encoder(semantic_emb)
        
        # Multi-Modal Feature Fusion (Concatenation + Fusion Network)
        # *** FUSION LOCATION ***
        combined_features = torch.cat([sensor_feat, vla_feat], dim=-1)
        fused_latents = self.fusion_network(combined_features)
        
        # Conditioned Dueling Streams
        state_values = self.value_stream(fused_latents)
        advantages = self.advantage_stream(fused_latents)
        
        # Mean-Normalized Aggregation Layer: Q(s, a; e_vla) = V(s, e_vla) + (A(s, a; e_vla) - mean(A))
        q_values = state_values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

# Alias for backward compatibility
DuelingDQN = VLAGuardedDuelingDQN

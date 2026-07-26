import numpy as np

class VLAGuard:
    """
    Simulated High-Level Vision-Language-Action (VLA) Guard.
    Simulates an asynchronous lightweight VLM (e.g., Prismatic / LLaVA-Phi)
    that analyzes environmental visual context and outputs semantic behavior tokens,
    which are projected into a dense continuous modulation vector e_vla (dim 64).
    """
    def __init__(self, semantic_dim: int = 64):
        self.semantic_dim = semantic_dim
        
        # Define semantic vocabulary tokens
        self.tokens = {
            "OPEN_WAREHOUSE": 0,
            "CROWDED_ROOM": 1,
            "HAZARDOUS_ZONE": 2
        }
        
        # Precompute deterministic orthogonal-like embedding vectors for each semantic token
        np.random.seed(100)
        self.embeddings = {
            "OPEN_WAREHOUSE": np.random.normal(loc=0.5, scale=0.1, size=semantic_dim).astype(np.float32),
            "CROWDED_ROOM": np.random.normal(loc=-0.5, scale=0.1, size=semantic_dim).astype(np.float32),
            "HAZARDOUS_ZONE": np.random.normal(loc=0.0, scale=0.8, size=semantic_dim).astype(np.float32)
        }
        
    def get_semantic_context(self, robot_x: float, robot_y: float, min_lidar_dist: float) -> tuple:
        """
        Analyzes environment spatial context and emits a high-level VLA semantic token
        and its corresponding continuous modulation embedding vector.
        
        Args:
            robot_x, robot_y: Current robot coordinates in 10x10 map.
            min_lidar_dist: Minimum clearance distance to nearest obstacle.
            
        Returns:
            (token_name, embedding_vector)
        """
        # Semantic Rule Reasoning:
        # 1. Hazardous zone: Near central obstacles or narrow clear distance (< 0.6m)
        if min_lidar_dist < 0.6:
            token = "HAZARDOUS_ZONE"
        # 2. Crowded room: Middle section of map with multiple obstacles (x in [2, 8], y in [2, 8])
        elif 2.5 <= robot_x <= 7.5 and 2.5 <= robot_y <= 7.5:
            token = "CROWDED_ROOM"
        # 3. Open warehouse: Outer clear boundaries
        else:
            token = "OPEN_WAREHOUSE"
            
        return token, self.embeddings[token]

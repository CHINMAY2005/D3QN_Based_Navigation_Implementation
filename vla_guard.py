import os
import numpy as np
import torch
from PIL import Image

class VLAGuard:
    """
    High-Level Vision-Language-Action (VLA) Guard.
    Analyzes environmental visual context (real RGB images from camera or dataset)
    and outputs semantic behavior tokens ("OPEN_WAREHOUSE", "CROWDED_ROOM", "HAZARDOUS_ZONE"),
    projecting them into a dense 64-dimensional continuous modulation vector e_vla.
    """
    def __init__(self, semantic_dim: int = 64, model_path: str = "checkpoints/vla_vision_encoder.pth"):
        self.semantic_dim = semantic_dim
        self.model_path = model_path
        
        # Vocabulary Mapping: ID -> Token String
        self.id_to_token = {
            0: "OPEN_WAREHOUSE",
            1: "CROWDED_ROOM",
            2: "HAZARDOUS_ZONE"
        }
        
        # Precomputed fallback embeddings
        np.random.seed(100)
        self.fallback_embeddings = {
            "OPEN_WAREHOUSE": np.random.normal(loc=0.5, scale=0.1, size=semantic_dim).astype(np.float32),
            "CROWDED_ROOM": np.random.normal(loc=-0.5, scale=0.1, size=semantic_dim).astype(np.float32),
            "HAZARDOUS_ZONE": np.random.normal(loc=0.0, scale=0.8, size=semantic_dim).astype(np.float32)
        }
        
        # Attempt loading PyTorch Vision Encoder model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = None
        
        if os.path.exists(self.model_path):
            try:
                from train_vla_vision_model import VLAVisionEncoder
                self.vision_model = VLAVisionEncoder(num_classes=3, semantic_dim=semantic_dim).to(self.device)
                self.vision_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.vision_model.eval()
                print(f"VLAGuard: Successfully loaded PyTorch Vision Encoder from {self.model_path}")
            except Exception as e:
                print(f"VLAGuard: Note - running in analytical/rule mode ({e})")
                self.vision_model = None

    def get_semantic_context_from_image(self, image_path_or_pil) -> tuple:
        """
        Processes a real RGB image using the trained PyTorch VLAVisionEncoder,
        predicts the semantic safety token, and extracts the 64-dim continuous embedding vector e_vla.
        
        Args:
            image_path_or_pil: Path to image file string or PIL Image object.
            
        Returns:
            (predicted_token_string, embedding_numpy_array_64dim)
        """
        if self.vision_model is not None:
            try:
                if isinstance(image_path_or_pil, str):
                    img = Image.open(image_path_or_pil).convert('RGB')
                else:
                    img = image_path_or_pil.convert('RGB')
                    
                img = img.resize((128, 128))
                img_arr = np.array(img, dtype=np.float32) / 255.0
                img_arr = np.transpose(img_arr, (2, 0, 1))
                img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                img_tensor = (img_tensor - 0.5) / 0.5
                
                with torch.no_grad():
                    logits, embedding = self.vision_model(img_tensor)
                    pred_class = torch.argmax(logits, dim=1).item()
                    token_str = self.id_to_token.get(pred_class, "CROWDED_ROOM")
                    emb_np = embedding.squeeze(0).cpu().numpy()
                    
                return token_str, emb_np
            except Exception as e:
                print(f"Error processing image in VLAGuard: {e}")
                
        token_str = "CROWDED_ROOM"
        return token_str, self.fallback_embeddings[token_str]

    def get_semantic_context(self, robot_x: float, robot_y: float, min_lidar_dist: float) -> tuple:
        """
        Analyzes environment spatial context and emits a high-level VLA semantic token.
        """
        if min_lidar_dist < 0.6:
            token = "HAZARDOUS_ZONE"
        elif 2.5 <= robot_x <= 7.5 and 2.5 <= robot_y <= 7.5:
            token = "CROWDED_ROOM"
        else:
            token = "OPEN_WAREHOUSE"
            
        return token, self.fallback_embeddings[token]

import os
import numpy as np
import torch
from PIL import Image

class VLAGuard:
    """
    High-Level Vision-Language-Action (VLA) Guard.
    Analyzes environmental visual context and detects specific physical objects
    [human, wall, chair, door, mirror, glass, shoe, phone, mouse, clear_path]
    to generate Object-Aware VLA Semantic Tokens ("OPEN_WAREHOUSE", "CROWDED_ROOM", "HAZARDOUS_ZONE")
    and 64-dimensional continuous modulation vectors e_vla.
    """
    def __init__(self, semantic_dim: int = 64, 
                 model_path: str = "checkpoints/vla_vision_encoder.pth",
                 object_model_path: str = "checkpoints/object_vla_encoder.pth"):
                 
        self.semantic_dim = semantic_dim
        self.model_path = model_path
        self.object_model_path = object_model_path
        
        self.id_to_token = {
            0: "OPEN_WAREHOUSE",
            1: "CROWDED_ROOM",
            2: "HAZARDOUS_ZONE"
        }
        
        self.object_classes = [
            "human", "wall", "chair", "door", "mirror",
            "glass", "shoe", "phone", "mouse", "clear_path"
        ]
        
        self.class_to_token = {
            "human": "HAZARDOUS_ZONE",
            "glass": "HAZARDOUS_ZONE",
            "mirror": "HAZARDOUS_ZONE",
            "wall": "CROWDED_ROOM",
            "chair": "CROWDED_ROOM",
            "shoe": "CROWDED_ROOM",
            "phone": "CROWDED_ROOM",
            "mouse": "CROWDED_ROOM",
            "door": "OPEN_WAREHOUSE",
            "clear_path": "OPEN_WAREHOUSE"
        }
        
        np.random.seed(100)
        self.fallback_embeddings = {
            "OPEN_WAREHOUSE": np.random.normal(loc=0.5, scale=0.1, size=semantic_dim).astype(np.float32),
            "CROWDED_ROOM": np.random.normal(loc=-0.5, scale=0.1, size=semantic_dim).astype(np.float32),
            "HAZARDOUS_ZONE": np.random.normal(loc=0.0, scale=0.8, size=semantic_dim).astype(np.float32)
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_model = None
        self.object_model = None
        
        # Load Object Vision Encoder if available
        if os.path.exists(self.object_model_path):
            try:
                from train_object_detector import ObjectAwareVLAVisionEncoder
                self.object_model = ObjectAwareVLAVisionEncoder(num_objects=len(self.object_classes), semantic_dim=semantic_dim).to(self.device)
                self.object_model.load_state_dict(torch.load(self.object_model_path, map_location=self.device))
                self.object_model.eval()
                print(f"VLAGuard: Successfully loaded Object Vision Encoder from {self.object_model_path}")
            except Exception as e:
                print(f"VLAGuard: Object Model note ({e})")
                self.object_model = None
                
        # Load Baseline Vision Encoder if available
        if os.path.exists(self.model_path) and self.object_model is None:
            try:
                from train_vla_vision_model import VLAVisionEncoder
                self.vision_model = VLAVisionEncoder(num_classes=3, semantic_dim=semantic_dim).to(self.device)
                self.vision_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.vision_model.eval()
                print(f"VLAGuard: Loaded baseline PyTorch Vision Encoder from {self.model_path}")
            except Exception as e:
                self.vision_model = None

    def detect_objects_and_get_context(self, image_path_or_pil) -> tuple:
        """
        Detects specific physical objects in image (human, wall, chair, door, mirror, glass, shoe, phone, mouse)
        and outputs: (detected_object_name, vla_token_string, continuous_embedding_vector_64dim).
        """
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert('RGB')
        else:
            img = image_path_or_pil.convert('RGB')
            
        img = img.resize((64, 64))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        if self.object_model is not None:
            with torch.no_grad():
                logits, embedding = self.object_model(img_tensor)
                pred_idx = torch.argmax(logits, dim=1).item()
                obj_name = self.object_classes[pred_idx]
                token_str = self.class_to_token.get(obj_name, "CROWDED_ROOM")
                emb_np = embedding.squeeze(0).cpu().numpy()
            return obj_name, token_str, emb_np
        elif self.vision_model is not None:
            with torch.no_grad():
                logits, embedding = self.vision_model(img_tensor)
                pred_class = torch.argmax(logits, dim=1).item()
                token_str = self.id_to_token.get(pred_class, "CROWDED_ROOM")
                emb_np = embedding.squeeze(0).cpu().numpy()
            return "obstacle", token_str, emb_np
            
        return "obstacle", "CROWDED_ROOM", self.fallback_embeddings["CROWDED_ROOM"]

    def get_semantic_context_from_image(self, image_path_or_pil) -> tuple:
        obj_name, token_str, emb_np = self.detect_objects_and_get_context(image_path_or_pil)
        return token_str, emb_np

    def get_semantic_context(self, robot_x: float, robot_y: float, min_lidar_dist: float) -> tuple:
        if min_lidar_dist < 0.6:
            token = "HAZARDOUS_ZONE"
        elif 2.5 <= robot_x <= 7.5 and 2.5 <= robot_y <= 7.5:
            token = "CROWDED_ROOM"
        else:
            token = "OPEN_WAREHOUSE"
            
        return token, self.fallback_embeddings[token]

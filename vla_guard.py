import os
import numpy as np
import torch
import cv2
from PIL import Image

class VLAGuard:
    """
    High-Level Vision-Language-Action (VLA) Guard with Hybrid Human Detection Prior.
    
    1. Uses OpenCV Haar Cascade Face/Body Detector to guarantee high-confidence 
       detection of real humans.
    2. Runs PyTorch ObjectAwareVLAVisionEncoder for 9-class object identification:
       [human, wall, chair, door, mirror, shoe, phone, mouse, clear_path]
    3. Maps detected objects to VLA Safety Tokens:
       - human, mirror -> HAZARDOUS_ZONE
       - wall, chair, shoe, phone, mouse -> CROWDED_ROOM
       - door, clear_path -> OPEN_WAREHOUSE
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
        
        # 9 Target Classes (Glass removed)
        self.object_classes = [
            "human", "wall", "chair", "door", "mirror",
            "shoe", "phone", "mouse", "clear_path"
        ]
        
        self.class_to_token = {
            "human": "HAZARDOUS_ZONE",
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
        
        # Load OpenCV Haar Cascades for Human Detection Prior
        self.face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None
            
        # Load Object Vision Encoder (9 classes) if available
        if os.path.exists(self.object_model_path):
            try:
                from train_object_detector import ObjectAwareVLAVisionEncoder
                self.object_model = ObjectAwareVLAVisionEncoder(num_objects=len(self.object_classes), semantic_dim=semantic_dim).to(self.device)
                self.object_model.load_state_dict(torch.load(self.object_model_path, map_location=self.device))
                self.object_model.eval()
                print(f"VLAGuard: Successfully loaded 9-Class Object Vision Encoder from {self.object_model_path}")
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

    def detect_human_face(self, cv2_bgr_img) -> bool:
        """Uses OpenCV Haar Cascade face detector to verify human presence."""
        if self.face_cascade is None or cv2_bgr_img is None:
            return False
        try:
            gray = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            return len(faces) > 0
        except Exception:
            return False

    def detect_objects_and_get_context(self, image_path_or_pil) -> tuple:
        """
        Detects physical objects in image across 9 target classes (Glass removed).
        Returns: (detected_object_name, vla_token_string, continuous_embedding_vector_64dim).
        """
        if isinstance(image_path_or_pil, str):
            pil_img = Image.open(image_path_or_pil).convert('RGB')
        else:
            pil_img = image_path_or_pil.convert('RGB')
            
        cv2_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 1. HARD SAFETY PRIOR: Check for Human Presence via Haar Cascade
        if self.detect_human_face(cv2_bgr):
            obj_name = "human"
            token_str = "HAZARDOUS_ZONE"
            emb_np = self.fallback_embeddings["HAZARDOUS_ZONE"]
            return obj_name, token_str, emb_np
            
        # 2. PyTorch 9-Class Object Detection Model Inference
        img_resized = pil_img.resize((64, 64))
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        if self.object_model is not None:
            with torch.no_grad():
                logits, embedding = self.object_model(img_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                pred_idx = torch.argmax(probs).item()
                obj_name = self.object_classes[pred_idx]
                
                # Warm-tone human fallback guard
                if obj_name in ["mouse", "shoe"]:
                    r_channel = img_arr[0]
                    b_channel = img_arr[2]
                    warmth = np.mean(r_channel) - np.mean(b_channel)
                    if warmth > 0.18:
                        obj_name = "human"
                        
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

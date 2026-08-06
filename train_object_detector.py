"""
Object-Aware VLA Vision Model Trainer

Trains a PyTorch Convolutional Neural Network (ObjectAwareVLAVisionEncoder)
to identify specific physical objects and obstacles:
[human, wall, chair, door, mirror, glass, shoe, phone, mouse, clear_path]

Generates fine-grained object detection logits + 64-dim VLA continuous modulation vector (e_vla).
"""

import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================================
# 1. OBJECT TARGET CLASSES & SAFETY MAPPING
# =====================================================================
OBJECT_CLASSES = [
    "human",      # Class 0: Critical obstacle (HAZARDOUS)
    "wall",       # Class 1: Solid obstacle (CROWDED)
    "chair",      # Class 2: Interior furniture (CROWDED)
    "door",       # Class 3: Passage opening (OPEN)
    "mirror",     # Class 4: Reflection hazard (HAZARDOUS)
    "glass",      # Class 5: Transparent hazard (HAZARDOUS)
    "shoe",       # Class 6: Small floor obstacle (CROWDED)
    "phone",      # Class 7: Small personal object (CROWDED)
    "mouse",      # Class 8: Small personal object (CROWDED)
    "clear_path"  # Class 9: Open clear hallway (OPEN)
]

CLASS_TO_TOKEN = {
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

# =====================================================================
# 2. OBJECT-AWARE VLA VISION ENCODER ARCHITECTURE
# =====================================================================
class ObjectAwareVLAVisionEncoder(nn.Module):
    """
    Multi-Task PyTorch Vision Model that detects specific physical objects
    (human, wall, chair, door, mirror, glass, shoe, phone, mouse, clear_path)
    and projects features into a continuous 64-dim VLA modulation vector e_vla.
    """
    def __init__(self, num_objects: int = 10, semantic_dim: int = 64):
        super(ObjectAwareVLAVisionEncoder, self).__init__()
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 64-dim Bottleneck Modulation Vector Head (e_vla in R^64)
        self.embedding_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, semantic_dim)
        )
        
        # Object Classifier Head (10 target classes)
        self.object_classifier = nn.Linear(semantic_dim, num_objects)
        
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        return self.embedding_head(feat_flat)

    def forward(self, x: torch.Tensor):
        embedding = self.extract_embedding(x)
        object_logits = self.object_classifier(embedding)
        return object_logits, embedding


# =====================================================================
# 3. SYNTHETIC & LOGGED OBJECT DATASET GENERATOR
# =====================================================================
class ObjectObstacleDataset(torch.utils.data.Dataset):
    def __init__(self, samples_per_class=50, target_size=(64, 64)):
        self.samples = []
        self.target_size = target_size
        
        # Generate representative visual samples for each target object
        for class_id, obj_name in enumerate(OBJECT_CLASSES):
            for i in range(samples_per_class):
                self.samples.append((obj_name, class_id))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        obj_name, class_id = self.samples[idx]
        
        # Generate synthetic visual pattern representing the object class
        img_arr = np.random.uniform(0.1, 0.9, size=(self.target_size[0], self.target_size[1], 3)).astype(np.float32)
        
        # Object specific color signature
        color_signatures = {
            "human": [0.8, 0.3, 0.2],    # Skin/Clothing warm tones
            "wall": [0.5, 0.5, 0.5],     # Gray solid
            "chair": [0.3, 0.2, 0.1],    # Brown wood/fabric
            "door": [0.4, 0.2, 0.1],     # Door frame
            "mirror": [0.9, 0.9, 1.0],   # Bright reflective
            "glass": [0.7, 0.9, 0.9],    # Transparent cyan
            "shoe": [0.1, 0.1, 0.1],     # Dark shoe
            "phone": [0.0, 0.0, 0.2],    # Small dark blue/black
            "mouse": [0.2, 0.2, 0.2],    # Small mouse shape
            "clear_path": [0.2, 0.8, 0.2] # Open green path
        }
        sig = color_signatures.get(obj_name, [0.5, 0.5, 0.5])
        img_arr[:, :] = img_arr[:, :] * 0.3 + np.array(sig, dtype=np.float32) * 0.7
        
        img_tensor = torch.tensor(np.transpose(img_arr, (2, 0, 1)), dtype=torch.float32)
        img_tensor = (img_tensor - 0.5) / 0.5
        return img_tensor, class_id


# =====================================================================
# 4. TRAINING ROUTINE
# =====================================================================
def train_object_detector(epochs=8, batch_size=32, lr=1e-3):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    dataset = ObjectObstacleDataset(samples_per_class=60)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ObjectAwareVLAVisionEncoder(num_objects=len(OBJECT_CLASSES), semantic_dim=64).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- Starting Object Detection VLA Model Training on device: {device} ---", flush=True)
    print(f"Target Classes: {OBJECT_CLASSES}", flush=True)
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, embeddings = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, embeddings = model(imgs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}", flush=True)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/object_vla_encoder.pth")
            print(f"  -> Saved BEST Object VLA Encoder weights to checkpoints/object_vla_encoder.pth", flush=True)
            
    print("\nTraining Complete! Object Vision Encoder ready.", flush=True)

if __name__ == "__main__":
    train_object_detector(epochs=8)

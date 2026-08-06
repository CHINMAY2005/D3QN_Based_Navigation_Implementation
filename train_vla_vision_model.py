import os
import sys
import csv
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================================
# 1. SEMANTIC CATEGORY MAPPING (67 MIT Indoor Classes -> 3 Safety Tokens)
# =====================================================================
SAFETY_TOKENS = {
    "OPEN_WAREHOUSE": 0,
    "CROWDED_ROOM": 1,
    "HAZARDOUS_ZONE": 2
}

# Mapping 67 MIT Indoor categories into safety tokens
CATEGORY_TO_SAFETY_TOKEN = {
    # 0: OPEN_WAREHOUSE / Clear Open Spaces
    "warehouse": 0, "garage": 0, "corridor": 0, "cloister": 0, "airport_inside": 0,
    "trainstation": 0, "mall": 0, "subway": 0, "inside_subway": 0, "inside_bus": 0, "hallway": 0,
    
    # 2: HAZARDOUS_ZONE / High Risk Bottlenecks
    "operating_room": 2, "hospitalroom": 2, "laboratorywet": 2, "computerroom": 2,
    "elevator": 2, "stairscase": 2, "prisoncell": 2, "winecellar": 2, "closet": 2, "poolinside": 2,
    
    # 1: CROWDED_ROOM / Default Cluttered Interior (all others)
}

def get_safety_label(category_name: str) -> int:
    cat_clean = category_name.lower()
    return CATEGORY_TO_SAFETY_TOKEN.get(cat_clean, 1) # Default to CROWDED_ROOM (1)


# =====================================================================
# 2. PYTORCH VLA VISION ENCODER ARCHITECTURE
# =====================================================================
class VLAVisionEncoder(nn.Module):
    """
    Convolutional Neural Network Vision Encoder for VLA Semantic Guarding.
    Extracts visual features from real images, projects them into a 
    64-dimensional continuous modulation embedding vector (e_vla),
    and classifies scenes into semantic safety tokens (OPEN_WAREHOUSE, CROWDED_ROOM, HAZARDOUS_ZONE).
    """
    def __init__(self, num_classes: int = 3, semantic_dim: int = 64):
        super(VLAVisionEncoder, self).__init__()
        
        # Feature Extractor Blocks
        self.features = nn.Sequential(
            # Block 1: Input 3 x 64 x 64 -> 32 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((4, 4)) # 128 x 4 x 4
        )
        
        # Continuous Bottleneck Embedding Projection (e_vla in R^64)
        self.embedding_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, semantic_dim) # Output: 64-dim continuous vector e_vla
        )
        
        # Semantic Safety Classification Head
        self.classifier = nn.Linear(semantic_dim, num_classes)
        
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the 64-dimensional continuous VLA modulation vector e_vla."""
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        embedding = self.embedding_head(feat_flat)
        return embedding

    def forward(self, x: torch.Tensor):
        embedding = self.extract_embedding(x)
        logits = self.classifier(embedding)
        return logits, embedding


# =====================================================================
# 3. PYTORCH DATASET LOADER FOR MIT INDOOR SCENE RECOGNITION
# =====================================================================
class MITIndoorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, target_size=(64, 64), max_samples_per_class=40):
        self.samples = []
        self.target_size = target_size
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
            
        categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        categories.sort()
        
        for cat in categories:
            cat_dir = os.path.join(root_dir, cat)
            label = get_safety_label(cat)
            
            img_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if max_samples_per_class:
                img_files = img_files[:max_samples_per_class]
                
            for fname in img_files:
                img_path = os.path.join(cat_dir, fname)
                self.samples.append((img_path, label))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size)
            
            # Normalize image array to [0, 1] and transpose to [C, H, W]
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_arr = np.transpose(img_arr, (2, 0, 1))
            
            # Standardize image normalization (mean=0.5, std=0.5)
            img_tensor = torch.tensor(img_arr, dtype=torch.float32)
            img_tensor = (img_tensor - 0.5) / 0.5
            
            return img_tensor, label
        except Exception as e:
            blank_tensor = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return blank_tensor, label


# =====================================================================
# 4. TRAINING & EVALUATION LOOP
# =====================================================================
def train_vla_vision_model(dataset_root: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    train_dir = os.path.join(dataset_root, "train")
    valid_dir = os.path.join(dataset_root, "valid")
    
    train_dataset = MITIndoorDataset(train_dir, target_size=(64, 64), max_samples_per_class=30)
    valid_dataset = MITIndoorDataset(valid_dir, target_size=(64, 64), max_samples_per_class=15)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLAVisionEncoder(num_classes=3, semantic_dim=64).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n--- Starting VLA Vision Encoder Training on device: {device} ---", flush=True)
    print(f"Dataset: MIT Indoor Scene Recognition ({len(train_dataset)} Train | {len(valid_dataset)} Val)", flush=True)
    print(f"Total Epochs: {epochs} | Batch Size: {batch_size} | Initial LR: {lr}\n", flush=True)
    
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    
    best_valid_acc = 0.0
    csv_log_path = os.path.join("plots", "vla_vision_training_metrics.csv")
    
    with open(csv_log_path, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Valid_Loss", "Valid_Accuracy", "LR"])
        
        for epoch in range(1, epochs + 1):
            # Training Phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
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
                
            epoch_train_loss = running_loss / total if total > 0 else 0.0
            epoch_train_acc = correct / total if total > 0 else 0.0
            
            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in valid_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits, embeddings = model(imgs)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    
            epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
            epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            train_losses.append(epoch_train_loss)
            valid_losses.append(epoch_val_loss)
            train_accs.append(epoch_train_acc)
            valid_accs.append(epoch_val_acc)
            
            writer.writerow([epoch, round(epoch_train_loss, 4), round(epoch_train_acc, 4), 
                             round(epoch_val_loss, 4), round(epoch_val_acc, 4), lr])
            
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2%} | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2%}", flush=True)
                  
            # Save Best Model Checkpoint
            if epoch_val_acc >= best_valid_acc:
                best_valid_acc = epoch_val_acc
                checkpoint_path = os.path.join("checkpoints", "vla_vision_encoder.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> Saved BEST VLA Vision Encoder weights (Val Acc: {best_valid_acc:.2%})", flush=True)
                
    print(f"\nSaved training telemetry to {csv_log_path}", flush=True)
    
    # Plotting Results
    plot_vla_vision_results(train_losses, valid_losses, train_accs, valid_accs, "plots")
    return model

def plot_vla_vision_results(train_losses, valid_losses, train_accs, valid_accs, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    
    # Loss Curve
    axes[0].plot(range(1, len(train_losses)+1), train_losses, color='crimson', linewidth=2, marker='o', label='Train Loss')
    axes[0].plot(range(1, len(valid_losses)+1), valid_losses, color='dodgerblue', linewidth=2, linestyle='--', marker='s', label='Valid Loss')
    axes[0].set_title('VLA Vision Encoder Loss (Cross Entropy)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Accuracy Curve
    axes[1].plot(range(1, len(train_accs)+1), train_accs, color='darkgreen', linewidth=2, marker='o', label='Train Accuracy')
    axes[1].plot(range(1, len(valid_accs)+1), valid_accs, color='darkorange', linewidth=2, linestyle='--', marker='s', label='Valid Accuracy')
    axes[1].set_title('VLA Semantic Token Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "vla_vision_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training curve plots to {plot_path}", flush=True)

if __name__ == "__main__":
    dataset_path = os.path.join("Datasets", "MIT Indoor Scene Recognition.v5-resized416by416_70-20-10split.folder")
    train_vla_vision_model(dataset_path, epochs=5, batch_size=64)

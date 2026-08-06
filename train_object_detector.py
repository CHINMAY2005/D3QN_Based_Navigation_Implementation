"""
Object-Aware VLA Vision Model Trainer (9 Classes, Excluding Glass)

Trains PyTorch ObjectAwareVLAVisionEncoder across 9 target physical obstacle classes:
[human, wall, chair, door, mirror, shoe, phone, mouse, clear_path]
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

OBJECT_CLASSES = [
    "human", "wall", "chair", "door", "mirror",
    "shoe", "phone", "mouse", "clear_path"
]

CLASS_TO_TOKEN = {
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

class ObjectAwareVLAVisionEncoder(nn.Module):
    def __init__(self, num_objects: int = 9, semantic_dim: int = 64):
        super(ObjectAwareVLAVisionEncoder, self).__init__()
        
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
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.embedding_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, semantic_dim)
        )
        
        self.object_classifier = nn.Linear(semantic_dim, num_objects)
        
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat_flat = feat.view(feat.size(0), -1)
        return self.embedding_head(feat_flat)

    def forward(self, x: torch.Tensor):
        embedding = self.extract_embedding(x)
        object_logits = self.object_classifier(embedding)
        return object_logits, embedding


class RealObjectDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="Datasets/Object_Obstacles", target_size=(64, 64), augment=True):
        self.samples = []
        self.target_size = target_size
        self.augment = augment
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found. Run download_and_build_dataset.py first.")
            
        for class_id, cls_name in enumerate(OBJECT_CLASSES):
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
                for fname in files:
                    self.samples.append((os.path.join(cls_dir, fname), class_id))
                    
        print(f"Dataset Loaded: {len(self.samples)} images across {len(OBJECT_CLASSES)} classes (Glass removed).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size)
            
            if self.augment and random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_arr = np.transpose(img_arr, (2, 0, 1))
            img_tensor = torch.tensor(img_arr, dtype=torch.float32)
            img_tensor = (img_tensor - 0.5) / 0.5
            return img_tensor, class_id
        except Exception:
            blank_tensor = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return blank_tensor, class_id


def train_object_detector(epochs=15, batch_size=32, lr=1e-3):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    dataset = RealObjectDataset(augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ObjectAwareVLAVisionEncoder(num_objects=len(OBJECT_CLASSES), semantic_dim=64).to(device)
    
    # 3x Loss Weight Penalty for Human Class to ensure human detection priority
    class_weights = torch.ones(len(OBJECT_CLASSES), dtype=torch.float32).to(device)
    class_weights[0] = 3.0 # Class 0: Human
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\n--- Starting 9-Class Object Vision Encoder Training (Glass Removed) ---", flush=True)
    print(f"Human Loss Weight: 3.0x | Classes ({len(OBJECT_CLASSES)}): {OBJECT_CLASSES}\n", flush=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    csv_log_path = os.path.join("plots", "object_vision_training_metrics.csv")
    with open(csv_log_path, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Epoch", "Train_Loss", "Train_Accuracy", "Valid_Loss", "Valid_Accuracy", "LR"])
        
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
                
            train_loss = running_loss / total if total > 0 else 0.0
            train_acc = correct / total if total > 0 else 0.0
            
            model.eval()
            v_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits, embeddings = model(imgs)
                    loss = criterion(logits, labels)
                    
                    v_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    
            val_loss = v_loss / val_total if val_total > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]['lr']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            writer.writerow([epoch, round(train_loss, 4), round(train_acc, 4), round(val_loss, 4), round(val_acc, 4), curr_lr])
            
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | LR: {curr_lr:.6f}", flush=True)
                  
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "checkpoints/object_vla_encoder.pth")
                print(f"  -> Saved BEST 9-Class Object VLA Encoder weights (Val Acc: {best_val_acc:.2%})", flush=True)
                
    plot_object_results(train_losses, val_losses, train_accs, val_accs, "plots")
    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2%}", flush=True)

def plot_object_results(train_losses, val_losses, train_accs, val_accs, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    
    axes[0].plot(range(1, len(train_losses)+1), train_losses, color='crimson', linewidth=2, marker='o', label='Train Loss')
    axes[0].plot(range(1, len(val_losses)+1), val_losses, color='dodgerblue', linewidth=2, linestyle='--', marker='s', label='Valid Loss')
    axes[0].set_title('9-Class Object Detection Loss (Glass Excluded)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].plot(range(1, len(train_accs)+1), train_accs, color='darkgreen', linewidth=2, marker='o', label='Train Accuracy')
    axes[1].plot(range(1, len(val_accs)+1), val_accs, color='darkorange', linewidth=2, linestyle='--', marker='s', label='Valid Accuracy')
    axes[1].set_title('9-Class Object Identification Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "object_vision_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved Object Detection curves plot to {plot_path}", flush=True)

if __name__ == "__main__":
    train_object_detector(epochs=15, batch_size=32)

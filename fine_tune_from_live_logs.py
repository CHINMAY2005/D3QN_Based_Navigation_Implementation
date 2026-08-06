"""
Continuous Fine-Tuning Script for PyTorch VLA Vision Encoder from Live Session Logs

Reads live camera frames and telemetry metadata logged to live_logs/ live_session_telemetry.csv
and fine-tunes the PyTorch VLAVisionEncoder (checkpoints/vla_vision_encoder.pth) to continually
improve model accuracy and semantic scene understanding in real physical environments.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from train_vla_vision_model import VLAVisionEncoder, SAFETY_TOKENS

class LiveSessionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file="live_logs/live_session_telemetry.csv", target_size=(64, 64)):
        self.samples = []
        self.target_size = target_size
        
        if not os.path.exists(csv_file):
            print(f"Warning: Telemetry log file '{csv_file}' not found.")
            return
            
        with open(csv_file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_path = row.get("Frame_Path")
                token_str = row.get("VLA_Token")
                label = SAFETY_TOKENS.get(token_str, 1) # Default to CROWDED_ROOM
                
                if frame_path and os.path.exists(frame_path):
                    self.samples.append((frame_path, label))
                    
        print(f"Loaded {len(self.samples)} logged live camera frames from {csv_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_size)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            img_arr = np.transpose(img_arr, (2, 0, 1))
            img_tensor = torch.tensor(img_arr, dtype=torch.float32)
            img_tensor = (img_tensor - 0.5) / 0.5
            return img_tensor, label
        except Exception as e:
            blank_tensor = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return blank_tensor, label

def fine_tune_vision_model(epochs=3, batch_size=16, lr=1e-4, checkpoint_path="checkpoints/vla_vision_encoder.pth"):
    os.makedirs("checkpoints", exist_ok=True)
    
    dataset = LiveSessionDataset()
    if len(dataset) == 0:
        print("No live camera session logs available for fine-tuning yet. Run a live camera session first!")
        return
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VLAVisionEncoder(num_classes=3, semantic_dim=64).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded existing VLA Vision Encoder weights from {checkpoint_path}")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n--- Starting Continual Fine-Tuning on Live Session Logs ({len(dataset)} Samples) ---")
    model.train()
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in loader:
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
            
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        print(f"Fine-Tune Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2%}")
        
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Successfully updated and saved fine-tuned VLA Vision Encoder weights to: {checkpoint_path}\n")

if __name__ == "__main__":
    fine_tune_vision_model(epochs=3)

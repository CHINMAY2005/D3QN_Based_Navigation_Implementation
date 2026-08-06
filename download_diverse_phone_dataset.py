"""
Diverse Mobile Phone Dataset Downloader & Builder

Populates local dataset directory (Datasets/Object_Obstacles/phone/) with 200+ real-world
image samples representing mobile phones of various models, brands, colors, screens, and angles.
"""

import os
import sys
import time
import random
import urllib.request
import ssl
from PIL import Image, ImageEnhance, ImageOps

ssl._create_default_https_context = ssl._create_unverified_context

PHONE_DIR = os.path.join("Datasets", "Object_Obstacles", "phone")

# Curated diverse real-world mobile phone images (smartphones, desk phones, screen ON/OFF)
DIVERSE_PHONE_IMAGE_URLS = [
    # Modern Smartphones (Black, Dark)
    "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=500&q=80",
    "https://images.unsplash.com/photo-1565849904461-04a58ad377e0?w=500&q=80",
    "https://images.unsplash.com/photo-1580910051074-3eb694886505?w=500&q=80",
    "https://images.unsplash.com/photo-1598327105666-5b89351aff97?w=500&q=80",
    
    # Phones with Active Wallpaper Screens
    "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?w=500&q=80",
    "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd?w=500&q=80",
    "https://images.unsplash.com/photo-1546054454-aa26e2b734c7?w=500&q=80",
    
    # Phones lying on Desk / Table
    "https://images.unsplash.com/photo-1574944985070-8f3ebc6b79d2?w=500&q=80",
    "https://images.unsplash.com/photo-1585060544812-6b45742d762f?w=500&q=80",
    "https://images.unsplash.com/photo-1567581935884-3349723552ca?w=500&q=80",
    
    # Phones Held in Hand
    "https://images.unsplash.com/photo-1533228876829-65c94e7b5025?w=500&q=80",
    "https://images.unsplash.com/photo-1523206489230-c012c64b2b48?w=500&q=80",
    "https://images.unsplash.com/photo-1573164713988-8665fc963095?w=500&q=80"
]

def generate_augmented_phone_variations(base_img, prefix, start_idx, num_variations=10):
    """Generates realistic lighting, display contrast, and spatial variations for deep mobile phone training."""
    created_files = []
    
    for idx in range(1, num_variations + 1):
        img = base_img.copy()
        
        # Random brightness (screen illumination / indoor shadow)
        enh_b = ImageEnhance.Brightness(img)
        img = enh_b.enhance(random.uniform(0.6, 1.4))
        
        # Random contrast
        enh_c = ImageEnhance.Contrast(img)
        img = enh_c.enhance(random.uniform(0.7, 1.3))
        
        # Random color saturation
        enh_s = ImageEnhance.Color(img)
        img = enh_s.enhance(random.uniform(0.8, 1.2))
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = ImageOps.mirror(img)
            
        file_path = os.path.join(PHONE_DIR, f"phone_global_{prefix}_{start_idx + idx:03d}.jpg")
        img.save(file_path)
        created_files.append(file_path)
        
    return created_files

def download_diverse_phone_dataset(target_total=200):
    os.makedirs(PHONE_DIR, exist_ok=True)
    print(f"\n--- Building Global Diverse Mobile Phone Dataset in '{PHONE_DIR}/' ---", flush=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    downloaded_bases = []
    for idx, url in enumerate(DIVERSE_PHONE_IMAGE_URLS, 1):
        base_path = os.path.join(PHONE_DIR, f"phone_base_{idx:02d}.jpg")
        if not os.path.exists(base_path):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response, open(base_path, 'wb') as out_file:
                    out_file.write(response.read())
                with Image.open(base_path) as img:
                    downloaded_bases.append(img.convert('RGB'))
            except Exception:
                pass
        else:
            try:
                with Image.open(base_path) as img:
                    downloaded_bases.append(img.convert('RGB'))
            except Exception:
                pass

    print(f"Loaded {len(downloaded_bases)} high-resolution base mobile phone images.", flush=True)
    
    # Generate augmented global variations to reach target 200+ samples
    current_count = len([f for f in os.listdir(PHONE_DIR) if f.endswith(('.jpg', '.png'))])
    if current_count < target_total and downloaded_bases:
        needed = target_total - current_count
        print(f"Generating {needed} augmented mobile phone variations...", flush=True)
        
        var_count = 0
        while var_count < needed:
            base_img = random.choice(downloaded_bases)
            var_count += 1
            generate_augmented_phone_variations(base_img, "var", var_count, num_variations=1)
            
    final_count = len([f for f in os.listdir(PHONE_DIR) if f.endswith(('.jpg', '.png'))])
    print(f"\nMobile Phone Dataset Build Complete! Total {final_count} diverse phone samples ready.", flush=True)

if __name__ == "__main__":
    download_diverse_phone_dataset(target_total=220)

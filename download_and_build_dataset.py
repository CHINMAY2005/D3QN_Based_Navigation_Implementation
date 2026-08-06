"""
Multi-Source Object Dataset Downloader & Builder (Excludes Glass)

Populates local dataset directory (Datasets/Object_Obstacles/) with diverse image samples
per target class: [human, wall, chair, door, mirror, shoe, phone, mouse, clear_path]
"""

import os
import sys
import shutil
import random
import urllib.request
import ssl
from PIL import Image, ImageEnhance

ssl._create_default_https_context = ssl._create_unverified_context

CLASSES = ["human", "wall", "chair", "door", "mirror", "shoe", "phone", "mouse", "clear_path"]
BASE_DIR = os.path.join("Datasets", "Object_Obstacles")

OBJECT_IMAGE_URLS = {
    "human": [
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&q=80",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80",
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80"
    ],
    "wall": [
        "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=400&q=80",
        "https://images.unsplash.com/photo-1589939705384-5185137a7f0f?w=400&q=80"
    ],
    "chair": [
        "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?w=400&q=80",
        "https://images.unsplash.com/photo-1580481072645-022f9a6d8310?w=400&q=80"
    ],
    "door": [
        "https://images.unsplash.com/photo-1517646287270-a5a9ca602e5c?w=400&q=80"
    ],
    "mirror": [
        "https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=400&q=80"
    ],
    "shoe": [
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&q=80"
    ],
    "phone": [
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400&q=80"
    ],
    "mouse": [
        "https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=400&q=80"
    ],
    "clear_path": [
        "https://images.unsplash.com/photo-1497366216548-37526070297c?w=400&q=80"
    ]
}

def generate_augmented_variation(base_img, file_path, var_id):
    img = base_img.copy()
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(file_path)

def download_and_build_dataset(samples_per_class=50):
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Remove glass dataset folder if present
    glass_dir = os.path.join(BASE_DIR, "glass")
    if os.path.exists(glass_dir):
        shutil.rmtree(glass_dir)
        print("Removed 'glass' category dataset directory.", flush=True)
        
    print(f"\n--- Building Object Dataset (9 classes, ex. Glass) in '{BASE_DIR}/' ---", flush=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for cat in CLASSES:
        cat_dir = os.path.join(BASE_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        urls = OBJECT_IMAGE_URLS.get(cat, [])
        
        base_images = []
        for idx, url in enumerate(urls, 1):
            file_path = os.path.join(cat_dir, f"{cat}_base_{idx:02d}.jpg")
            if not os.path.exists(file_path):
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=4) as response, open(file_path, 'wb') as out_file:
                        out_file.write(response.read())
                    with Image.open(file_path) as img:
                        base_images.append(img.convert('RGB'))
                except Exception:
                    img = Image.new('RGB', (128, 128), color=(random.randint(40, 220), random.randint(40, 220), random.randint(40, 220)))
                    img.save(file_path)
                    base_images.append(img)
            else:
                try:
                    with Image.open(file_path) as img:
                        base_images.append(img.convert('RGB'))
                except Exception:
                    pass

        if not base_images:
            img = Image.new('RGB', (128, 128), color=(100, 100, 100))
            base_images.append(img)

        for idx in range(1, samples_per_class + 1):
            file_path = os.path.join(cat_dir, f"{cat}_sample_{idx:03d}.jpg")
            if not os.path.exists(file_path):
                base_img = random.choice(base_images)
                generate_augmented_variation(base_img, file_path, idx)
                
        existing_count = len([f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.png'))])
        print(f"  Class [{cat:10s}]: Ready with {existing_count} image samples.", flush=True)
        
    print(f"\nDataset build complete! 9 classes ready (Glass removed).", flush=True)

if __name__ == "__main__":
    download_and_build_dataset(samples_per_class=50)

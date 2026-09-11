"""
Dynamic Dataset Manager & Class Configurator

Handles:
1. Creation of individual dataset subdirectories under Datasets/Object_Obstacles/<class_name>/
2. Registration of custom Object vs Path categories and their VLA safety tokens.
3. Automated sample image population & augmentation for new custom classes.
"""

import os
import json
import shutil
import random
import urllib.request
import ssl
from PIL import Image, ImageDraw, ImageEnhance

ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = os.path.join("Datasets", "Object_Obstacles")
CONFIG_PATH = os.path.join("checkpoints", "class_config.json")

DEFAULT_CLASSES = ["human", "wall", "chair", "door", "shoe", "phone", "clear_path"]
DEFAULT_TOKENS = {
    "human": "HAZARDOUS_ZONE",
    "wall": "CROWDED_ROOM",
    "chair": "CROWDED_ROOM",
    "door": "OPEN_WAREHOUSE",
    "shoe": "CROWDED_ROOM",
    "phone": "CROWDED_ROOM",
    "clear_path": "OPEN_WAREHOUSE"
}
DEFAULT_TYPES = {
    "human": "object",
    "wall": "object",
    "chair": "object",
    "door": "path",
    "shoe": "object",
    "phone": "object",
    "clear_path": "path"
}

def load_class_config():
    """Loads class config JSON or builds it dynamically from dataset directories."""
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(BASE_DIR, exist_ok=True)
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                return config
        except Exception:
            pass
            
    # Fallback / Discovery mode
    discovered_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    classes = list(set(DEFAULT_CLASSES + discovered_dirs))
    classes.sort()
    
    tokens = {}
    types = {}
    for c in classes:
        tokens[c] = DEFAULT_TOKENS.get(c, "CROWDED_ROOM")
        types[c] = DEFAULT_TYPES.get(c, "object")
        
    config = {
        "classes": classes,
        "class_to_token": tokens,
        "class_types": types
    }
    save_class_config(config)
    return config

def save_class_config(config):
    """Saves class configuration JSON."""
    os.makedirs("checkpoints", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def generate_custom_sample_images(class_name, cat_type, cat_dir, count=50):
    """Generates synthetic/augmented seed image samples for newly added custom class."""
    os.makedirs(cat_dir, exist_ok=True)
    existing = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png'))]
    if len(existing) >= count:
        return len(existing)
        
    # Generate distinct geometric/textured seed patterns based on category type
    seed_images = []
    base_colors = {
        "object": [(180, 50, 50), (50, 180, 50), (50, 50, 180), (200, 150, 30), (120, 40, 160)],
        "path": [(60, 180, 180), (100, 200, 100), (220, 220, 100), (180, 180, 220)]
    }
    palette = base_colors.get(cat_type, base_colors["object"])
    
    for i in range(5):
        img = Image.new("RGB", (128, 128), color=random.choice(palette))
        draw = ImageDraw.Draw(img)
        if cat_type == "path":
            # Draw perspective path lines
            draw.polygon([(40, 128), (88, 128), (70, 30), (58, 30)], fill=(240, 240, 240))
        else:
            # Draw bounding box/shape representing custom object
            draw.rectangle([20, 20, 108, 108], outline=(255, 255, 255), width=4)
            draw.ellipse([35, 35, 93, 93], fill=(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)))
        seed_images.append(img)
        
    for idx in range(1, count + 1):
        file_path = os.path.join(cat_dir, f"{class_name}_sample_{idx:03d}.jpg")
        if not os.path.exists(file_path):
            base_img = random.choice(seed_images).copy()
            enhancer = ImageEnhance.Brightness(base_img)
            base_img = enhancer.enhance(random.uniform(0.6, 1.4))
            enhancer = ImageEnhance.Contrast(base_img)
            base_img = enhancer.enhance(random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                base_img = base_img.transpose(Image.FLIP_LEFT_RIGHT)
            base_img.save(file_path)
            
    final_count = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png'))])
    return final_count

def add_custom_class(class_name, category_type="object", token=None, samples_count=50):
    """
    Creates dataset directory for custom class, generates samples, and updates class config.
    category_type: "object" or "path"
    token: "HAZARDOUS_ZONE", "CROWDED_ROOM", or "OPEN_WAREHOUSE"
    """
    clean_name = class_name.strip().lower().replace(" ", "_")
    if not clean_name:
        raise ValueError("Class name cannot be empty.")
        
    config = load_class_config()
    
    if token is None:
        if category_type.lower() == "path":
            token = "OPEN_WAREHOUSE"
        else:
            token = "CROWDED_ROOM"
            
    cat_dir = os.path.join(BASE_DIR, clean_name)
    sample_num = generate_custom_sample_images(clean_name, category_type.lower(), cat_dir, count=samples_count)
    
    if clean_name not in config["classes"]:
        config["classes"].append(clean_name)
        config["classes"].sort()
        
    config["class_to_token"][clean_name] = token
    config["class_types"][clean_name] = category_type.lower()
    
    save_class_config(config)
    print(f"Successfully registered class '{clean_name}' [{category_type.upper()}] with token [{token}] and {sample_num} samples.")
    return clean_name, token, sample_num

def get_dataset_summary():
    """Returns summary of all active classes in dataset."""
    config = load_class_config()
    summary = []
    
    for c in config["classes"]:
        cdir = os.path.join(BASE_DIR, c)
        count = 0
        if os.path.exists(cdir):
            count = len([f for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.png'))])
        summary.append({
            "name": c,
            "token": config["class_to_token"].get(c, "CROWDED_ROOM"),
            "type": config["class_types"].get(c, "object"),
            "samples": count,
            "path": cdir
        })
    return summary

if __name__ == "__main__":
    summary = get_dataset_summary()
    print("Dataset Summary:")
    for item in summary:
        print(f"  Class [{item['name']:12s}] | Type: {item['type']:6s} | Token: {item['token']:15s} | Samples: {item['samples']}")

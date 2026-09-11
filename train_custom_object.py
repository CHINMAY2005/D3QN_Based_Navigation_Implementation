"""
CLI Helper for Dynamic Custom Object & Path Model Training

Usage:
  python train_custom_object.py --names "laptop, water_bottle" --type object --samples 50 --train
  python train_custom_object.py --names "corridor_path" --type path --token OPEN_WAREHOUSE --train
"""

import sys
import os
import argparse
from dynamic_dataset_manager import add_custom_class, get_dataset_summary
from train_object_detector import train_rigorous_object_detector

def main():
    parser = argparse.ArgumentParser(description="Train VLA Model on Custom Objects or Paths")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated or newline-separated custom object/path names")
    parser.add_argument("--type", type=str, choices=["object", "path"], default="object", help="Category type: 'object' (hazard/obstacle) or 'path' (clear route)")
    parser.add_argument("--token", type=str, choices=["HAZARDOUS_ZONE", "CROWDED_ROOM", "OPEN_WAREHOUSE"], default=None, help="VLA Safety Token Override")
    parser.add_argument("--samples", type=int, default=50, help="Number of seed sample images per category")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--train", action="store_true", help="Automatically trigger model retraining after directory creation")
    
    args = parser.parse_args()
    
    raw_names = [n.strip() for n in args.names.replace("\n", ",").split(",") if n.strip()]
    if not raw_names:
        print("Error: No valid object/path names provided.")
        sys.exit(1)
        
    print(f"\n=======================================================")
    print(f"       DYNAMIC VLA CUSTOM CLASS TRAINER STUDIO       ")
    print(f"=======================================================\n")
    print(f"Category Selection: {args.type.upper()}")
    print(f"Custom Input Names ({len(raw_names)}): {', '.join(raw_names)}")
    
    registered = []
    for name in raw_names:
        c_name, token, count = add_custom_class(name, category_type=args.type, token=args.token, samples_count=args.samples)
        registered.append((c_name, token, count))
        
    print("\nUpdated Dataset Summary:")
    summary = get_dataset_summary()
    for item in summary:
        print(f"  Class [{item['name']:15s}] | Type: {item['type']:6s} | Token: {item['token']:15s} | Samples: {item['samples']}")
        
    if args.train:
        print(f"\n--- Initiating Model Retraining across all {len(summary)} dataset classes ---")
        train_rigorous_object_detector(epochs=args.epochs, batch_size=32)
        print("\nRetraining Complete! Model weights saved to checkpoints/object_vla_encoder.pth")
    else:
        print("\nDirectories created & registered! Run with '--train' flag to trigger model training.")

if __name__ == "__main__":
    main()

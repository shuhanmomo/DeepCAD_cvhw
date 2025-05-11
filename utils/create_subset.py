import json
import random
import os
import numpy as np
from pathlib import Path


def load_split_file(path):
    """Load the split file containing train/validation/test splits"""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"],
    }


def save_split_file(data, path):
    """Save the split file in the same format as the original"""
    output_data = {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"],
    }
    with open(path, "w") as f:
        json.dump(output_data, f, indent=2)


def create_subset(split_data, train_size=6000, val_size=1000, test_size=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Create subset for each split
    subset = {
        "train": random.sample(split_data["train"], train_size),
        "validation": random.sample(split_data["validation"], val_size),
        "test": random.sample(split_data["test"], test_size),
    }

    return subset


def main():
    # Load original split
    split_path = os.path.join("data", "train_val_test_split.json")
    split_data = load_split_file(split_path)

    # Print original sizes
    print("Original dataset sizes:")
    for split in ["train", "validation", "test"]:
        print(f"{split}: {len(split_data[split])}")

    # Create subset
    subset = create_subset(split_data)

    # Save subset split file
    subset_path = os.path.join("data", "subset_split.json")
    save_split_file(subset, subset_path)

    # Print subset sizes
    print("\nSubset sizes:")
    for split in ["train", "validation", "test"]:
        print(f"{split}: {len(subset[split])}")

    # Create directories for subset images
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join("data", f"img_{split}_subset"), exist_ok=True)

    print("\nNext steps:")
    print("1. Run the following commands to render images for each split:")
    for split in ["train", "validation", "test"]:
        print(f"python utils/render_for_clip.py --split {split} --subset")


if __name__ == "__main__":
    main()

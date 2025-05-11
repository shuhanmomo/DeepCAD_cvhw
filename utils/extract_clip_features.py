import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time


def load_clip_model(device):
    """Load CLIP model and return model and preprocess function"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # Set to evaluation mode
    return model, preprocess


def process_model_views_batch(model_dir, model, preprocess, device, batch_size=24):
    """Process all views of a single model in batches and return list of CLIP features"""
    view_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".jpg")])
    if not view_files:
        return None

    # Process in batches
    all_features = []
    for i in range(0, len(view_files), batch_size):
        batch_files = view_files[i : i + batch_size]
        batch_images = []

        # Load and preprocess batch of images
        for view_file in batch_files:
            image_path = os.path.join(model_dir, view_file)
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image)
            batch_images.append(image_input)

        # Stack batch and move to device
        batch_tensor = torch.stack(batch_images).to(device)

        # Extract features
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
            batch_features = batch_features.cpu().numpy()

        all_features.extend(batch_features)

    # Return list of features (one per view)
    if all_features:
        return [feat.tolist() for feat in all_features]
    return None


def load_existing_results(output_path):
    """Load existing results if available"""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return json.load(f)
    return {"train": [], "validation": [], "test": []}


def save_results(results, output_path):
    """Save results to JSON file efficiently"""
    # Create backup of existing file if it exists
    if os.path.exists(output_path):
        backup_path = output_path + ".backup"
        os.replace(output_path, backup_path)

    # Save new results using a temporary file
    temp_path = output_path + ".temp"
    with open(temp_path, "w") as f:
        json.dump(results, f)

    # Atomic replace of the old file with the new one
    os.replace(temp_path, output_path)

    # Remove backup if save was successful
    if os.path.exists(backup_path):
        os.remove(backup_path)


def process_split(
    split_dir, model, preprocess, device, batch_size=24, save_interval=100
):
    """Process all models in a split and return list of results"""
    results = []
    output_path = os.path.join("data", "CLIP_feats.json")

    # Load existing results to check what's already processed
    existing_results = load_existing_results(output_path)
    split_name = os.path.basename(split_dir).split("_")[
        1
    ]  # e.g., "train" from "img_train_subset"
    processed_ids = {item["id"] for item in existing_results.get(split_name, [])}

    print(f"Found {len(processed_ids)} already processed models in {split_name} split")

    # Get all model directories (parent_id/child_id)
    model_dirs = []
    for parent_id in os.listdir(split_dir):
        parent_path = os.path.join(split_dir, parent_id)
        if os.path.isdir(parent_path):
            for child_id in os.listdir(parent_path):
                child_path = os.path.join(parent_path, child_id)
                if os.path.isdir(child_path) and any(
                    f.endswith(".jpg") for f in os.listdir(child_path)
                ):
                    model_id = f"{parent_id}/{child_id}"
                    if model_id not in processed_ids:  # Only add unprocessed models
                        model_dirs.append((child_path, model_id))

    print(f"Found {len(model_dirs)} models remaining to process in {split_name} split")

    # Process each model directory
    for i, (model_dir, model_id) in enumerate(
        tqdm(model_dirs, desc=f"Processing {os.path.basename(split_dir)}")
    ):
        # Process all views of this model
        clip_feats = process_model_views_batch(
            model_dir, model, preprocess, device, batch_size
        )

        if clip_feats is not None:
            results.append(
                {
                    "id": model_id,
                    "clip_feats": clip_feats,  # List of features, one per view
                }
            )

            # Save intermediate results periodically
            if (i + 1) % save_interval == 0:
                # Update the current split's results in memory
                existing_results[split_name] = (
                    existing_results.get(split_name, []) + results
                )
                # Save all results
                save_results(existing_results, output_path)
                print(f"\nSaved intermediate results after processing {i + 1} models")
                # Clear the results list after saving
                results = []

    # Save any remaining results
    if results:
        existing_results[split_name] = existing_results.get(split_name, []) + results
        save_results(existing_results, output_path)
        print(f"\nSaved final results for {split_name} split")

    return existing_results[split_name]


def estimate_processing_time(split_dir, batch_size=24):
    """Estimate processing time based on number of models and views"""
    total_models = 0
    for parent_id in os.listdir(split_dir):
        parent_path = os.path.join(split_dir, parent_id)
        if os.path.isdir(parent_path):
            for child_id in os.listdir(parent_path):
                child_path = os.path.join(parent_path, child_id)
                if os.path.isdir(child_path) and any(
                    f.endswith(".jpg") for f in os.listdir(child_path)
                ):
                    total_models += 1

    # Rough estimate: 0.1 seconds per batch of 24 images
    estimated_time = (total_models * 0.1) / 60  # Convert to minutes
    return total_models, estimated_time


def main():
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = load_clip_model(device)

    # Load existing results if available
    output_path = os.path.join("data", "CLIP_feats.json")
    results = load_existing_results(output_path)

    # Process each split
    for split in ["train", "validation", "test"]:
        print(f"\nProcessing {split} split...")
        split_dir = os.path.join("data", f"img_{split}_subset")

        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue

        # Estimate processing time
        total_models, est_time = estimate_processing_time(split_dir)
        print(f"Found {total_models} total models in {split} split")
        print(f"Estimated processing time: {est_time:.1f} minutes")

        # Process the split
        start_time = time.time()
        results[split] = process_split(
            split_dir, model, preprocess, device, save_interval=100
        )
        end_time = time.time()

        print(f"Processed {len(results[split])} models in {split} split")
        print(f"Actual processing time: {(end_time - start_time)/60:.1f} minutes")

        # Save results after each split
        save_results(results, output_path)
        print(f"Saved results after completing {split} split")

    print("Done!")


if __name__ == "__main__":
    main()

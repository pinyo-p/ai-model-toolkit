#!/usr/bin/env python3
"""Merge sharded safetensors files into a single file.

Usage:
    python scripts/merge_shards.py <shard_folder> [output_path]

Example:
    python scripts/merge_shards.py /Users/Shared/AI/models/text_encoder/z-image
    python scripts/merge_shards.py /path/to/shards /path/to/output.safetensors
"""
import sys
import os
import json

def merge_shards(folder, output_path=None):
    folder = os.path.abspath(folder)
    
    # Find index.json
    index_file = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        # Try single file
        safetensors = [f for f in os.listdir(folder) if f.endswith('.safetensors')]
        if len(safetensors) == 1:
            print(f"Only one safetensors file found: {safetensors[0]}")
            print("No need to merge.")
            return
        elif len(safetensors) == 0:
            print("No safetensors files found!")
            return
        else:
            print("Multiple safetensors files but no index.json found.")
            print("Files:", safetensors)
            return

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    
    # Group tensors by shard file
    shard_files = {}
    for tensor_name, shard_file in weight_map.items():
        if shard_file not in shard_files:
            shard_files[shard_file] = []
        shard_files[shard_file].append(tensor_name)

    print(f"Found {len(shard_files)} shards with {len(weight_map)} tensors total")

    # Load and merge
    from safetensors.torch import load_file, save_file
    
    all_tensors = {}
    for shard_file, tensor_names in sorted(shard_files.items()):
        shard_path = os.path.join(folder, shard_file)
        print(f"Loading {shard_file} ({len(tensor_names)} tensors)...")
        shard_data = load_file(shard_path)
        for name in tensor_names:
            if name in shard_data:
                all_tensors[name] = shard_data[name]
    
    print(f"Merged {len(all_tensors)} tensors")

    if output_path is None:
        output_path = os.path.join(folder, "model.safetensors")
    
    print(f"Saving to {output_path}...")
    save_file(all_tensors, output_path)
    
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Done! Output size: {size_gb:.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    folder = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    merge_shards(folder, output)

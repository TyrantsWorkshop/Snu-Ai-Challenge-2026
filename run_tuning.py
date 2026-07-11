import subprocess
import os
import sys
import torch
import pandas as pd

# Define the 4 tuning configurations
configs = [
    {
        "name": "1_frozen_baseline",
        "args": [
            "--freeze_vision",
            "--freeze_text",
            "--epochs", "8",
            "--batch_size", "16",
            "--num_workers", "2",
            "--output_dir", "./checkpoints/run_1_frozen"
        ]
    },
    {
        "name": "2_regularized_finetune",
        "args": [
            "--backbone_lr", "3e-6",
            "--head_lr", "5e-5",
            "--weight_decay", "0.1",
            "--epochs", "8",
            "--batch_size", "8",
            "--num_workers", "2",
            "--output_dir", "./checkpoints/run_2_regularized"
        ]
    },
    {
        "name": "3_frozen_heavyweights",
        "args": [
            "--vision_backbone", "vit_base_patch16_224",
            "--text_backbone", "bert-base-uncased",
            "--embed_dim", "768",
            "--num_heads", "12",
            "--freeze_vision",
            "--freeze_text",
            "--epochs", "8",
            "--batch_size", "16",
            "--num_workers", "2",
            "--output_dir", "./checkpoints/run_3_heavyweights"
        ]
    },
    {
        "name": "4_deberta_semantic",
        "args": [
            "--text_backbone", "microsoft/deberta-v3-small",
            "--backbone_lr", "5e-6",
            "--head_lr", "8e-5",
            "--weight_decay", "0.05",
            "--epochs", "8",
            "--batch_size", "8",
            "--num_workers", "2",
            "--output_dir", "./checkpoints/run_4_deberta"
        ]
    }
]

def run_experiment(config):
    name = config["name"]
    args = config["args"]
    cmd = [sys.executable, "train.py"] + args

    # Check if checkpoint already exists to skip training
    ckpt_path = os.path.join(args[args.index("--output_dir") + 1], "best_model.pt")
    if os.path.exists(ckpt_path):
        try:
            import torch
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            val_acc = checkpoint.get("val_exact_acc", None)
            if val_acc is not None and isinstance(val_acc, float):
                print(f"\n>>> Found existing checkpoint for {name}. Skipping training. Best Val Exact Match: {val_acc:.4f}\n")
                return val_acc
        except Exception:
            pass

    print("=" * 60)
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run train.py and stream output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    # Extract results from saved checkpoint
    ckpt_path = os.path.join(args[args.index("--output_dir") + 1], "best_model.pt")
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            val_acc = checkpoint.get("val_exact_acc", "Unknown")
            print(f"\n>>> Experiment {name} completed! Best Val Exact Match: {val_acc:.4f}\n")
            return val_acc
        except Exception as e:
            print(f"\n>>> Experiment {name} completed, but error reading checkpoint: {e}\n")
            return "Error"
    else:
        print(f"\n>>> Experiment {name} failed (no checkpoint saved).\n")
        return "Failed"

def main():
    results = {}
    for config in configs:
        val_acc = run_experiment(config)
        results[config["name"]] = val_acc
        
    print("\n" + "=" * 60)
    print("ALL TUNING EXPERIMENTS COMPLETE - SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Experiment Name':<30} | {'Best Val Exact Match':<20}")
    print("-" * 60)
    for name, acc in results.items():
        if isinstance(acc, float):
            print(f"{name:<30} | {acc:.4%}")
        else:
            print(f"{name:<30} | {acc}")
    print("=" * 60)

if __name__ == "__main__":
    main()

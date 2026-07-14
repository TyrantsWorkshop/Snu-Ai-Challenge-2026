import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_cached import CachedDataset, CachedFrameOrderModel, hungarian_predictions

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading cached test features...")
    img_feats = torch.load(os.path.join(args.cache_dir, "test_img_feats.pt"), map_location="cpu")
    txt_feats = torch.load(os.path.join(args.cache_dir, "test_txt_feats.pt"), map_location="cpu")
    attn_masks = torch.load(os.path.join(args.cache_dir, "test_attn_masks.pt"), map_location="cpu")
    test_ids_df = pd.read_csv(os.path.join(args.cache_dir, "test_ids.csv"))
    
    test_ds = CachedDataset(img_feats, txt_feats, attn_masks, labels=None)
    test_loader = DataLoader(test_dataset=test_ds, batch_size=args.batch_size, shuffle=False)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    model = CachedFrameOrderModel(
        embed_dim=ckpt_args["embed_dim"],
        num_fusion_layers=ckpt_args["num_fusion_layers"],
        num_heads=ckpt_args["num_heads"],
        dropout=ckpt_args["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting cached"):
            img_feat = batch["img_feat"].to(device)
            txt_feat = batch["txt_feat"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            logits = model(img_feat, txt_feat, attn_mask)
            logits_np = logits.detach().cpu().numpy()
            preds = hungarian_predictions(logits_np)

            all_preds.extend((preds + 1).tolist())  # back to 1-indexed

    out_df = pd.DataFrame({
        "Id": test_ids_df["Id"],
        "Answer": [str(p) for p in all_preds],
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, default="./data/cached_features_clip")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/cached_best_model.pt")
    p.add_argument("--output_csv", type=str, default="./submission.csv")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_inference(args)

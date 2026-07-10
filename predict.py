import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dataset import FrameOrderDataset
from model import FrameOrderModel
from train import build_transform, hungarian_predictions  # reuse the same logic as training


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]

    tokenizer = AutoTokenizer.from_pretrained(ckpt_args["text_backbone"])
    transform = build_transform(ckpt_args["image_size"])

    test_dataset = FrameOrderDataset(
        csv_path=args.test_csv,
        img_root=args.test_img_root,
        image_transform=transform,
        tokenizer=tokenizer,
        max_len=ckpt_args["max_len"],
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = FrameOrderModel(
        vision_backbone=ckpt_args["vision_backbone"],
        text_backbone=ckpt_args["text_backbone"],
        embed_dim=ckpt_args["embed_dim"],
        num_fusion_layers=ckpt_args["num_fusion_layers"],
        num_heads=ckpt_args["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            imgs = batch["imgs"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            ids = batch["id"]

            logits = model(imgs, input_ids, attn)
            logits_np = logits.detach().cpu().numpy()
            preds = hungarian_predictions(logits_np)  # [B,4], 0-indexed slots

            all_ids.extend(list(ids))
            all_preds.extend((preds + 1).tolist())  # back to 1-indexed to match Answer format

    # NOTE: double check sample_submission.csv for the exact expected column
    # names / string formatting and adjust below if it differs.
    out_df = pd.DataFrame({
        "Id": all_ids,
        "Answer": [str(p) for p in all_preds],
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    p.add_argument("--test_csv", type=str, default="./data/snuaichallenge_data/test.csv")
    p.add_argument("--test_img_root", type=str, default="./data/snuaichallenge_data/test")
    p.add_argument("--output_csv", type=str, default="./submission.csv")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_inference(args)

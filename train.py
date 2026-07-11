import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import AutoTokenizer
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dataset import FrameOrderDataset
from model import FrameOrderModel


def build_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Alias for backward compatibility (e.g. imported in predict.py)
build_transform = build_val_transform


def hungarian_predictions(logits):
    """
    logits: [B,4,4] numpy array, row=input frame, col=candidate slot.
    Returns: [B,4] predicted 0-indexed slot for each input, guaranteed to be
    a valid permutation per sample.
    """
    preds = np.zeros((logits.shape[0], 4), dtype=np.int64)
    for b in range(logits.shape[0]):
        cost = -logits[b]  # linear_sum_assignment minimizes cost
        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind is already [0,1,2,3] sorted; col_ind gives assigned slot per row
        preds[b, row_ind] = col_ind
    return preds


def evaluate(model, loader, device):
    model.eval()
    exact_matches = 0
    total = 0
    correct_positions = 0
    total_positions = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["imgs"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()  # [B,4], 0-indexed

            logits = model(imgs, input_ids, attn)
            logits_np = logits.detach().cpu().numpy()
            preds = hungarian_predictions(logits_np)

            exact_matches += (preds == labels).all(axis=1).sum()
            correct_positions += (preds == labels).sum()
            total += labels.shape[0]
            total_positions += labels.size

    exact_match_acc = exact_matches / total
    position_acc = correct_positions / total_positions
    return exact_match_acc, position_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
    df = pd.read_csv(args.train_csv)
    n = len(df)
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_transform = build_train_transform(args.image_size)
    val_transform = build_val_transform(args.image_size)

    train_ds = FrameOrderDataset(
        csv_path_or_df=train_df,
        img_root=args.train_img_root,
        image_transform=train_transform,
        tokenizer=tokenizer,
        max_len=args.max_len,
        is_train=True,
    )
    val_ds = FrameOrderDataset(
        csv_path_or_df=val_df,
        img_root=args.train_img_root,
        image_transform=val_transform,
        tokenizer=tokenizer,
        max_len=args.max_len,
        is_train=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = FrameOrderModel(
        vision_backbone=args.vision_backbone,
        text_backbone=args.text_backbone,
        embed_dim=args.embed_dim,
        num_fusion_layers=args.num_fusion_layers,
        num_heads=args.num_heads,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
    ).to(device)

    # Differential learning rates: pretrained backbones get a smaller LR,
    # the from-scratch fusion module + head get a larger LR.
    backbone_params = list(model.vision_encoder.parameters()) + list(model.text_encoder.parameters())
    new_params = (
        list(model.vis_proj.parameters()) + list(model.text_proj.parameters()) +
        list(model.fusion.parameters()) + list(model.fusion_norm.parameters()) +
        list(model.slot_head.parameters()) + list(model.input_id_embed.parameters()) +
        [model.frame_type_embed, model.text_type_embed]
    )

    optimizer = torch.optim.AdamW([
        {"params": [p for p in backbone_params if p.requires_grad], "lr": args.backbone_lr},
        {"params": [p for p in new_params if p.requires_grad], "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Determine AMP precision and dtype
    amp_dtype = torch.float16
    use_bf16 = False
    if device.type == "cuda" and args.amp:
        if "deberta" in args.text_backbone.lower() or (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()):
            amp_dtype = torch.bfloat16
            use_bf16 = True
            print("Using bfloat16 for Automatic Mixed Precision (AMP) training.")

    use_scaler = (device.type == "cuda" and args.amp and not use_bf16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            imgs = batch["imgs"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)  # [B,4]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
                logits = model(imgs, input_ids, attn)  # [B,4,4]
                # cross entropy per row (per input frame), summed over the 4 rows
                loss = 0.0
                for i in range(4):
                    loss = loss + criterion(logits[:, i, :], labels[:, i])
                loss = loss / 4.0

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        val_exact_acc, val_pos_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: val_exact_match_acc={val_exact_acc:.4f}  val_position_acc={val_pos_acc:.4f}")

        if val_exact_acc > best_val_acc:
            best_val_acc = val_exact_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "val_exact_acc": val_exact_acc,
            }, best_ckpt_path)
            print(f"  -> saved new best checkpoint (val_exact_match_acc={val_exact_acc:.4f})")

    print(f"Training complete. Best val exact-match accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="./data/snuaichallenge_data/train.csv")
    p.add_argument("--train_img_root", type=str, default="./data/snuaichallenge_data/train")
    p.add_argument("--output_dir", type=str, default="./checkpoints")

    p.add_argument("--vision_backbone", type=str, default="vit_small_patch16_224")
    p.add_argument("--text_backbone", type=str, default="distilbert-base-uncased")
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--num_fusion_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--freeze_vision", action="store_true")
    p.add_argument("--freeze_text", action="store_true")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_len", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    train(args)

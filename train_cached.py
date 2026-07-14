import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

class CachedDataset(Dataset):
    def __init__(self, img_feats, txt_feats, attn_masks, labels=None):
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        self.attn_masks = attn_masks
        self.labels = labels

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        sample = {
            "img_feat": self.img_feats[idx],
            "txt_feat": self.txt_feats[idx],
            "attn_mask": self.attn_masks[idx]
        }
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        return sample

class CachedFrameOrderModel(nn.Module):
    def __init__(self, embed_dim=512, num_fusion_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.vis_proj = nn.Linear(768, embed_dim)
        self.text_proj = nn.Linear(512, embed_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        self.input_id_embed = nn.Embedding(4, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_fusion_layers)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        self.slot_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
        )

    def forward(self, img_feat, txt_feat, attn_mask):
        B = img_feat.shape[0]
        vis_feat = self.vis_proj(img_feat)  # [B, 4, embed_dim]
        text_feat = self.text_proj(txt_feat)  # [B, 77, embed_dim]
        
        key_padding_mask = (attn_mask == 0)
        cross_out, _ = self.cross_attention(
            query=vis_feat,
            key=text_feat,
            value=text_feat,
            key_padding_mask=key_padding_mask
        )
        vis_feat = self.cross_norm(vis_feat + cross_out)
        
        ids = torch.arange(4, device=img_feat.device).unsqueeze(0).expand(B, 4)
        vis_feat = vis_feat + self.input_id_embed(ids)
        
        fused = self.fusion(vis_feat)
        fused = self.fusion_norm(fused)
        
        logits = self.slot_head(fused)  # [B, 4, 4]
        return logits

def hungarian_predictions(logits):
    preds = np.zeros((logits.shape[0], 4), dtype=np.int64)
    for b in range(logits.shape[0]):
        cost = -logits[b]
        row_ind, col_ind = linear_sum_assignment(cost)
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
            img_feat = batch["img_feat"].to(device)
            txt_feat = batch["txt_feat"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["label"].numpy()  # [B, 4]

            logits = model(img_feat, txt_feat, attn_mask)
            logits_np = logits.detach().cpu().numpy()
            preds = hungarian_predictions(logits_np)

            exact_matches += (preds == labels).all(axis=1).sum()
            correct_positions += (preds == labels).sum()
            total += labels.shape[0]
            total_positions += labels.size

    exact_match_acc = exact_matches / total
    position_acc = correct_positions / total_positions
    return exact_match_acc, position_acc

def train_cached(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load cache
    print("Loading cached CLIP features...")
    img_feats = torch.load(os.path.join(args.cache_dir, "train_img_feats.pt"), map_location="cpu")
    txt_feats = torch.load(os.path.join(args.cache_dir, "train_txt_feats.pt"), map_location="cpu")
    attn_masks = torch.load(os.path.join(args.cache_dir, "train_attn_masks.pt"), map_location="cpu")
    labels = torch.load(os.path.join(args.cache_dir, "train_labels.pt"), map_location="cpu")
    
    n = len(img_feats)
    print(f"Loaded {n} samples.")

    # Splits
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = CachedDataset(
        img_feats[train_idx],
        txt_feats[train_idx],
        attn_masks[train_idx],
        labels[train_idx]
    )
    val_ds = CachedDataset(
        img_feats[val_idx],
        txt_feats[val_idx],
        attn_masks[val_idx],
        labels[val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = CachedFrameOrderModel(
        embed_dim=args.embed_dim,
        num_fusion_layers=args.num_fusion_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "cached_best_model.pt")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            img_feat = batch["img_feat"].to(device)
            txt_feat = batch["txt_feat"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            target_labels = batch["label"].to(device)  # [B, 4]

            optimizer.zero_grad()
            logits = model(img_feat, txt_feat, attn_mask)  # [B, 4, 4]
            
            loss = 0.0
            for i in range(4):
                loss = loss + criterion(logits[:, i, :], target_labels[:, i])
            loss = loss / 4.0

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
    p.add_argument("--cache_dir", type=str, default="./data/cached_features_clip")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_fusion_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    
    p.add_argument("--batch_size", type=int, default=64)  # Can use large batch size since features are cached!
    p.add_argument("--epochs", type=int, default=50)       # Can run more epochs since it's extremely fast!
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    train_cached(args)

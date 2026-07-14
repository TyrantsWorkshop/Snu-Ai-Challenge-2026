import os
import ast
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from model import resolve_model_path
from train import build_val_transform

class ExtractionDataset(Dataset):
    def __init__(self, csv_path, img_root, transform, tokenizer, max_len=77, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _load_image(self, vid_id, filename):
        path = os.path.join(self.img_root, str(vid_id), str(filename))
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_id = row["Id"]
        sentence = str(row["Sentence"])

        # Load 4 images
        imgs = [self._load_image(vid_id, row[f"Input_{i}"]) for i in range(1, 5)]
        imgs = torch.stack(imgs, dim=0)  # [4, C, H, W]

        # Tokenize sentence
        enc = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        sample = {
            "imgs": imgs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": vid_id
        }

        if self.is_train:
            answer = ast.literal_eval(row["Answer"]) if isinstance(row["Answer"], str) else list(row["Answer"])
            labels = torch.tensor([a - 1 for a in answer], dtype=torch.long)
            sample["labels"] = labels

        return sample

def extract_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "openai/clip-vit-base-patch16"
    resolved_model = resolve_model_path(model_name)

    print("Loading CLIP models...")
    tokenizer = CLIPTokenizer.from_pretrained(resolved_model, local_files_only=True)
    vision_encoder = CLIPVisionModel.from_pretrained(resolved_model, local_files_only=True).to(device)
    text_encoder = CLIPTextModel.from_pretrained(resolved_model, local_files_only=True).to(device)

    vision_encoder.eval()
    text_encoder.eval()

    # Paths
    train_csv = "./data/snuaichallenge_data/train.csv"
    train_img = "./data/snuaichallenge_data/train"
    test_csv = "./data/snuaichallenge_data/test.csv"
    test_img = "./data/snuaichallenge_data/test"
    output_dir = "./data/cached_features_clip"
    os.makedirs(output_dir, exist_ok=True)

    # Use CLIP normalization for feature extraction
    transform = build_val_transform(image_size=224, is_clip=True)

    # 1. Process Train Set
    print("\n=== Extracting Train Features ===")
    train_ds = ExtractionDataset(train_csv, train_img, transform, tokenizer, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)

    all_img_feats = []
    all_txt_feats = []
    all_attn_masks = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Train features"):
            imgs = batch["imgs"].to(device)  # [B, 4, C, H, W]
            input_ids = batch["input_ids"].to(device)  # [B, 77]
            attn_mask = batch["attention_mask"].to(device)  # [B, 77]
            labels = batch["labels"]  # [B, 4]
            ids = batch["id"]

            B, N, C, H, W = imgs.shape
            flat_imgs = imgs.view(B * N, C, H, W)

            # Vision forward
            vis_out = vision_encoder(pixel_values=flat_imgs)
            vis_feat = vis_out.pooler_output.view(B, N, -1).cpu()  # [B, 4, 768]

            # Text forward
            text_out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            text_feat = text_out.last_hidden_state.cpu()  # [B, 77, 512]

            all_img_feats.append(vis_feat)
            all_txt_feats.append(text_feat)
            all_attn_masks.append(attn_mask.cpu())
            all_labels.append(labels)
            all_ids.extend(ids)

    # Concat and save train
    torch.save(torch.cat(all_img_feats, dim=0), os.path.join(output_dir, "train_img_feats.pt"))
    torch.save(torch.cat(all_txt_feats, dim=0), os.path.join(output_dir, "train_txt_feats.pt"))
    torch.save(torch.cat(all_attn_masks, dim=0), os.path.join(output_dir, "train_attn_masks.pt"))
    torch.save(torch.cat(all_labels, dim=0), os.path.join(output_dir, "train_labels.pt"))
    pd.DataFrame({"Id": all_ids}).to_csv(os.path.join(output_dir, "train_ids.csv"), index=False)
    print("Train features cached successfully!")

    # 2. Process Test Set
    print("\n=== Extracting Test Features ===")
    test_ds = ExtractionDataset(test_csv, test_img, transform, tokenizer, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_img_feats_test = []
    all_txt_feats_test = []
    all_attn_masks_test = []
    all_ids_test = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test features"):
            imgs = batch["imgs"].to(device)  # [B, 4, C, H, W]
            input_ids = batch["input_ids"].to(device)  # [B, 77]
            attn_mask = batch["attention_mask"].to(device)  # [B, 77]
            ids = batch["id"]

            B, N, C, H, W = imgs.shape
            flat_imgs = imgs.view(B * N, C, H, W)

            # Vision forward
            vis_out = vision_encoder(pixel_values=flat_imgs)
            vis_feat = vis_out.pooler_output.view(B, N, -1).cpu()  # [B, 4, 768]

            # Text forward
            text_out = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            text_feat = text_out.last_hidden_state.cpu()  # [B, 77, 512]

            all_img_feats_test.append(vis_feat)
            all_txt_feats_test.append(text_feat)
            all_attn_masks_test.append(attn_mask.cpu())
            all_ids_test.extend(ids)

    # Concat and save test
    torch.save(torch.cat(all_img_feats_test, dim=0), os.path.join(output_dir, "test_img_feats.pt"))
    torch.save(torch.cat(all_txt_feats_test, dim=0), os.path.join(output_dir, "test_txt_feats.pt"))
    torch.save(torch.cat(all_attn_masks_test, dim=0), os.path.join(output_dir, "test_attn_masks.pt"))
    pd.DataFrame({"Id": all_ids_test}).to_csv(os.path.join(output_dir, "test_ids.csv"), index=False)
    print("Test features cached successfully!")

if __name__ == "__main__":
    extract_features()

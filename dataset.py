"""
Dataset loader for the frame-reordering task.

Expected directory layout (as described in the challenge):

    train.csv
    test.csv
    train/<Id>/<Id>_xxx.jpg   (4 files per Id, referenced by Input_1..Input_4)
    test/<Id>/<Id>_xxx.jpg

train.csv columns : Id, Sentence, Input_1, Input_2, Input_3, Input_4, No_ordering, Answer
test.csv  columns : Id, Sentence, Input_1, Input_2, Input_3, Input_4

Answer format: a stringified list like "[2, 4, 3, 1]" meaning
    Input_1 belongs at temporal position 2
    Input_2 belongs at temporal position 4
    Input_3 belongs at temporal position 3
    Input_4 belongs at temporal position 1
"""

import os
import ast
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def parse_answer(raw):
    """Answer column may already be a list, or a string like '[2, 4, 3, 1]'."""
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return list(raw)


class FrameOrderDataset(Dataset):
    def __init__(self, csv_path_or_df, img_root, image_transform, tokenizer,
                 max_len=48, is_train=True):
        if isinstance(csv_path_or_df, str):
            self.df = pd.read_csv(csv_path_or_df)
        else:
            self.df = csv_path_or_df
        self.img_root = img_root
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

        # sanity check the expected columns exist
        required = ["Id", "Sentence", "Input_1", "Input_2", "Input_3", "Input_4"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV {csv_path} is missing expected columns: {missing}")

    def __len__(self):
        return len(self.df)

    def _load_image(self, vid_id, filename):
        path = os.path.join(self.img_root, str(vid_id), str(filename))
        img = Image.open(path).convert("RGB")
        return self.image_transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_id = row["Id"]
        sentence = str(row["Sentence"])

        imgs = [self._load_image(vid_id, row[f"Input_{i}"]) for i in range(1, 5)]
        imgs = torch.stack(imgs, dim=0)  # [4, C, H, W]

        enc = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        sample = {
            "imgs": imgs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": vid_id,
        }

        if self.is_train:
            answer = parse_answer(row["Answer"])  # e.g. [2, 4, 3, 1], 1-indexed
            labels = torch.tensor([a - 1 for a in answer], dtype=torch.long)  # 0-indexed slots
            sample["labels"] = labels

        return sample

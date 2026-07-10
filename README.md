# Frame Reordering Model

Predicts the correct temporal order of 4 shuffled video frames given a caption.

## Architecture

```
4 frames --> ViT (timm, pretrained)      --> per-frame embeddings [B,4,D]
caption  --> BERT/DistilBERT (HF, pretrained) --> caption embedding [B,1,D]
      |
      v
concat with learnable "which Input_k" identity tags
      |
      v
small Transformer encoder (fusion / reasoning module, trained from scratch)
      |
      v
linear head --> [B,4,4] affinity matrix
   (row i = Input_i, column j = candidate temporal position j)
```

- **Training loss**: cross-entropy on each row against the ground-truth position (from `Answer`).
- **Inference**: the raw per-row argmax can produce invalid outputs (two frames claiming
  the same slot). Instead we run the **Hungarian algorithm**
  (`scipy.optimize.linear_sum_assignment`) on the affinity matrix, which finds the
  globally best *valid* permutation. This directly improves exact-match accuracy over
  naive argmax, at zero training cost.
- Why separate ViT + BERT instead of CLIP: CLIP's text tower is trained for
  image-caption matching, not for parsing temporal/sequencing language ("first",
  "then", "before", "after"). A general language model plus a task-specific fusion
  transformer (trained from scratch on your data) is better suited to actually reason
  about those cues relative to the visual content.

## Setup

```bash
pip install -r requirements.txt
```

Since inference/training must run internet-blocked, **pre-download the pretrained
weights once while you still have internet access** so they're cached locally
(`~/.cache/huggingface` and `~/.cache/torch`):

```bash
python -c "import timm; timm.create_model('vit_small_patch16_224', pretrained=True)"
python -c "from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('distilbert-base-uncased'); \
AutoTokenizer.from_pretrained('distilbert-base-uncased')"
```

Both of these checkpoints were released well before the competition's cutoff, but
double-check the release date of whatever backbone you finally pick and note it in
your report as the rules require.

## Directory layout expected

```
data/
  snuaichallenge_data/
    train.csv
    test.csv
    train/<Id>/<Id>_xxx.jpg   (4 images per Id)
    test/<Id>/<Id>_xxx.jpg
```

## Train

```bash
python train.py \
  --train_csv ./data/snuaichallenge_data/train.csv \
  --train_img_root ./data/snuaichallenge_data/train \
  --output_dir ./checkpoints \
  --batch_size 16 \
  --epochs 15
```

This does a reproducible 90/10 train/val split (seeded), reports both exact-match
accuracy (all 4 correct) and per-position accuracy each epoch, and saves the best
checkpoint (by exact-match accuracy) to `checkpoints/best_model.pt`.

## Predict / build submission

```bash
python predict.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test_csv ./data/snuaichallenge_data/test.csv \
  --test_img_root ./data/snuaichallenge_data/test \
  --output_csv ./submission.csv
```

**Important**: check `sample_submission.csv` for the exact expected column names /
string formatting and adjust the last few lines of `predict.py` if they differ from
what's assumed here (`Id`, `Answer` as a stringified list like `[2, 4, 3, 1]`).

## Tuning for your hardware (RTX 4050 / 6GB VRAM)

- If you're CPU-only or have limited VRAM: keep `vision_backbone=vit_small_patch16_224`,
  `text_backbone=distilbert-base-uncased`, lower `--batch_size`, and consider
  `--freeze_vision --freeze_text` (only trains the fusion module + projections,
  much faster, usually gives up most but not all accuracy).
- If you have a strong GPU (16GB+): try `vision_backbone=vit_base_patch16_224` and
  `text_backbone=roberta-base` or `bert-base-uncased`, and drop the freeze flags to
  fully fine-tune.
- LoRA fine-tuning on the backbones is explicitly allowed by the rules and is a good
  middle ground if you want to fine-tune a larger backbone cheaply — not implemented
  here, but straightforward to add via the `peft` library if useful.

## Compliance notes (per the rules you shared)

- Single model, no ensembling — matches "no combining inference results of different
  models" rule.
- Uses only the provided `train.csv`/`train/` data for learning; no external datasets.
- Uses only pretrained open-source weights (verify release date < May 31, 2026 for
  whichever checkpoint you finalize).
- No external commercial API calls anywhere in this code.
- Runs fully offline once the pretrained weights are cached locally beforehand.

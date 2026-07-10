"""
Frame-ordering model.

Pipeline:
  4 frames --> ViT (timm)      --> per-frame embeddings [B,4,D]
  caption  --> BERT (HF)       --> caption embedding     [B,1,D]
  concat + learnable "which input slot" id embeddings
      --> small Transformer encoder (fusion / reasoning module, trained from scratch)
      --> linear head --> [B,4,4] affinity matrix
          (row i = Input_i, column j = candidate temporal position j)

Training: cross-entropy per row against the true temporal position.
Inference: Hungarian algorithm (scipy) on the affinity matrix to guarantee
           a valid permutation (see predict.py).
"""

import torch
import torch.nn as nn
import timm
from transformers import AutoModel


class FrameOrderModel(nn.Module):
    def __init__(
        self,
        vision_backbone="vit_small_patch16_224",
        text_backbone="distilbert-base-uncased",
        embed_dim=384,
        num_fusion_layers=4,
        num_heads=6,
        dropout=0.1,
        freeze_vision=False,
        freeze_text=False,
    ):
        super().__init__()

        # ---- Vision encoder ----
        self.vision_encoder = timm.create_model(vision_backbone, pretrained=True, num_classes=0)
        vis_dim = self.vision_encoder.num_features
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # ---- Text encoder ----
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        text_dim = self.text_encoder.config.hidden_size
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # ---- Projections into shared space ----
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # Learnable embedding telling the fusion module "this token is Input_k".
        # This is NOT the answer / temporal order -- it's just an identity tag
        # so the model can tell the 4 frame tokens apart and later report a
        # per-input-slot prediction.
        self.input_id_embed = nn.Embedding(4, embed_dim)

        # modality-type embeddings (frame vs. text token)
        self.frame_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Fusion / reasoning transformer (trained from scratch) ----
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

        # ---- Output head: per-frame logits over 4 candidate positions ----
        self.slot_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4),
        )

    def encode_images(self, imgs):
        # imgs: [B, 4, C, H, W]
        B, N, C, H, W = imgs.shape
        flat = imgs.view(B * N, C, H, W)
        feats = self.vision_encoder(flat)          # [B*4, vis_dim]
        feats = self.vis_proj(feats).view(B, N, -1)  # [B,4,embed_dim]
        return feats

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]-equivalent token
        return self.text_proj(cls)            # [B, embed_dim]

    def forward(self, imgs, input_ids, attention_mask):
        B = imgs.shape[0]

        vis_feat = self.encode_images(imgs)  # [B,4,D]
        ids = torch.arange(4, device=imgs.device).unsqueeze(0).expand(B, 4)
        vis_feat = vis_feat + self.input_id_embed(ids) + self.frame_type_embed

        text_feat = self.encode_text(input_ids, attention_mask).unsqueeze(1)  # [B,1,D]
        text_feat = text_feat + self.text_type_embed

        seq = torch.cat([vis_feat, text_feat], dim=1)  # [B,5,D]
        fused = self.fusion(seq)
        fused = self.fusion_norm(fused)

        frame_tokens = fused[:, :4, :]  # drop the text token, keep 4 frame tokens
        logits = self.slot_head(frame_tokens)  # [B,4,4]
        return logits

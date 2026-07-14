"""
Frame-ordering model.

Pipeline:
  - Baseline:
    4 frames --> ViT (timm)      --> per-frame embeddings [B,4,D]
    caption  --> BERT (HF)       --> caption embedding     [B,1,D]
    concat + learnable "which input slot" id embeddings
        --> small Transformer encoder (fusion / reasoning module, trained from scratch)
        --> linear head --> [B,4,4] affinity matrix
  - CLIP-based:
    4 frames --> CLIP Vision Encoder --> [B, 4, 768]
    caption  --> CLIP Text Encoder   --> [B, L, 512]
    Cross-Attention (Query: Vision, Key/Value: Text)
    Add learnable "which input slot" id embeddings
        --> small Transformer encoder (fusion / reasoning module, trained from scratch)
        --> linear head --> [B,4,4] affinity matrix
"""

import os
import torch
import torch.nn as nn
import timm
from transformers import AutoModel, CLIPVisionModel, CLIPTextModel


def resolve_model_path(model_name):
    """
    Resolves a model name to its offline cache path if present.
    Ensures the model runs internet-blocked on local snapshots.
    """
    import os
    if model_name == "openai/clip-vit-base-patch16":
        p = "G:/SnuAi/hf_cache/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
        if os.path.exists(p):
            return p
    elif model_name == "openai/clip-vit-base-patch32":
        p = "G:/SnuAi/hf_cache/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        if os.path.exists(p):
            return p
        # Check default user cache location too
        p_user = os.path.expanduser("~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
        if os.path.exists(p_user):
            return p_user
    elif model_name == "microsoft/deberta-v3-small":
        p = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--deberta-v3-small/snapshots/a36c739020e01763fe789b4b85e2df55d6180012")
        if os.path.exists(p):
            return p
    elif model_name == "distilbert-base-uncased":
        p = os.path.expanduser("~/.cache/huggingface/hub/models--distilbert-base-uncased/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be")
        if os.path.exists(p):
            return p
    elif model_name == "bert-base-uncased":
        p = os.path.expanduser("~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
        if os.path.exists(p):
            return p
    return model_name


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

        # Check if we should use CLIP architecture
        self.is_clip = "clip" in vision_backbone.lower() or "clip" in text_backbone.lower()

        if self.is_clip:
            print(f"Using CLIP Architecture with Cross-Attention Fusion.")
            print(f"Vision backbone: {vision_backbone}, Text backbone: {text_backbone}")

            # ---- CLIP Vision encoder ----
            resolved_vision = resolve_model_path(vision_backbone)
            self.vision_encoder = CLIPVisionModel.from_pretrained(resolved_vision, local_files_only=True)
            vis_dim = self.vision_encoder.config.hidden_size  # Typically 768 for ViT-B
            if freeze_vision:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = False

            # ---- CLIP Text encoder ----
            resolved_text = resolve_model_path(text_backbone)
            self.text_encoder = CLIPTextModel.from_pretrained(resolved_text, local_files_only=True)
            text_dim = self.text_encoder.config.hidden_size  # Typically 512
            if freeze_text:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False

            # ---- Projections ----
            self.vis_proj = nn.Linear(vis_dim, embed_dim)
            self.text_proj = nn.Linear(text_dim, embed_dim)

            # ---- Cross-Attention Fusion ----
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_norm = nn.LayerNorm(embed_dim)

            # ---- Slot Head ID Tag ----
            self.input_id_embed = nn.Embedding(4, embed_dim)

            # ---- Fusion Transformer (Self-Attention reasoning over frames) ----
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

            # ---- Output head ----
            self.slot_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 4),
            )

        else:
            # ---- Standard Baseline Architecture ----
            # ---- Vision encoder ----
            self.vision_encoder = timm.create_model(vision_backbone, pretrained=True, num_classes=0)
            vis_dim = self.vision_encoder.num_features
            if freeze_vision:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = False

            # ---- Text encoder ----
            resolved_text = resolve_model_path(text_backbone)
            self.text_encoder = AutoModel.from_pretrained(resolved_text)
            text_dim = self.text_encoder.config.hidden_size
            if freeze_text:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False

            # ---- Projections into shared space ----
            self.vis_proj = nn.Linear(vis_dim, embed_dim)
            self.text_proj = nn.Linear(text_dim, embed_dim)

            # Learnable identity tags
            self.input_id_embed = nn.Embedding(4, embed_dim)

            # Modality type embeddings
            self.frame_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.text_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

            # ---- Fusion transformer (trained from scratch) ----
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

            # ---- Output head ----
            self.slot_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 4),
            )

    def encode_images(self, imgs):
        B, N, C, H, W = imgs.shape
        flat = imgs.view(B * N, C, H, W)
        feats = self.vision_encoder(flat)            # [B*4, vis_dim]
        feats = self.vis_proj(feats).view(B, N, -1)  # [B,4,embed_dim]
        return feats

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]         # [B, text_dim]
        return self.text_proj(cls)                  # [B, embed_dim]

    def forward(self, imgs, input_ids, attention_mask):
        B = imgs.shape[0]

        if self.is_clip:
            B, N, C, H, W = imgs.shape
            flat_imgs = imgs.view(B * N, C, H, W)
            
            # 1. Vision Feature Extraction
            vis_out = self.vision_encoder(pixel_values=flat_imgs)
            vis_feat = vis_out.pooler_output             # [B*4, 768]
            vis_feat = self.vis_proj(vis_feat).view(B, N, -1)  # [B, 4, embed_dim]

            # 2. Text Feature Extraction
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = text_out.last_hidden_state       # [B, L, 512]
            text_feat = self.text_proj(text_feat)        # [B, L, embed_dim]

            # 3. Cross-Attention alignment
            # Build key_padding_mask: True where attention_mask is 0 (padded tokens)
            key_padding_mask = (attention_mask == 0)
            cross_out, _ = self.cross_attention(
                query=vis_feat,
                key=text_feat,
                value=text_feat,
                key_padding_mask=key_padding_mask
            )
            vis_feat = self.cross_norm(vis_feat + cross_out)

            # 4. Identity tags
            ids = torch.arange(4, device=imgs.device).unsqueeze(0).expand(B, 4)
            vis_feat = vis_feat + self.input_id_embed(ids)

            # 5. Temporal Reasoning
            fused = self.fusion(vis_feat)                # [B, 4, embed_dim]
            fused = self.fusion_norm(fused)

            logits = self.slot_head(fused)               # [B, 4, 4]
            return logits

        else:
            # Baseline Forward
            vis_feat = self.encode_images(imgs)          # [B,4,embed_dim]
            ids = torch.arange(4, device=imgs.device).unsqueeze(0).expand(B, 4)
            vis_feat = vis_feat + self.input_id_embed(ids) + self.frame_type_embed

            text_feat = self.encode_text(input_ids, attention_mask).unsqueeze(1)  # [B,1,embed_dim]
            text_feat = text_feat + self.text_type_embed

            seq = torch.cat([vis_feat, text_feat], dim=1) # [B,5,embed_dim]
            fused = self.fusion(seq)
            fused = self.fusion_norm(fused)

            frame_tokens = fused[:, :4, :]
            logits = self.slot_head(frame_tokens)         # [B,4,4]
            return logits

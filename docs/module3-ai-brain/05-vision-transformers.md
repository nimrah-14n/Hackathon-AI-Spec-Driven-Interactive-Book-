---
sidebar_position: 5
title: "Vision Transformers"
---

# Vision Transformers

## Learning Outcomes
By the end of this chapter, you will be able to:
- Understand the fundamental architecture of Vision Transformers (ViTs)
- Implement Vision Transformer models for robotic vision tasks
- Apply ViTs to various computer vision problems in robotics
- Evaluate the advantages and limitations of ViTs compared to CNNs
- Design efficient ViT architectures for resource-constrained robotic systems

## Introduction to Vision Transformers

Vision Transformers (ViTs) represent a paradigm shift in computer vision, applying the transformer architecture originally designed for natural language processing to visual data. Unlike traditional convolutional neural networks (CNNs) that rely on local receptive fields and spatial hierarchies, ViTs treat images as sequences of patches, enabling global attention mechanisms that can capture long-range dependencies in visual scenes.

### The Vision Transformer Revolution

The introduction of Vision Transformers marked a significant departure from the dominance of CNNs in computer vision:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Reshape to patches
        x = x.view(
            batch_size,
            channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size
        )

        # Rearrange to (batch_size, n_patches, patch_dim)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, self.n_patches, -1)

        # Project to embedding dimension
        x = self.projection(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super(VisionTransformer, self).__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )

        # Dropout
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Take the class token output
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x
```

### Why Vision Transformers for Robotics?

Vision Transformers offer several advantages for robotic applications:

#### Global Context Understanding
- **Long-range Dependencies**: Capture relationships between distant image regions
- **Scene Understanding**: Better comprehension of entire visual scenes
- **Multi-object Relations**: Understand interactions between multiple objects

#### Scalability
- **Performance Scaling**: Improve with larger datasets and models
- **Transfer Learning**: Effective pre-training on large datasets
- **Multi-task Learning**: Single model for multiple vision tasks

```
Vision Transformer Architecture for Robotics
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Image   │───→│  Patch &        │───→│  Transformer    │
│   (RGB, Depth,  │    │  Embedding      │    │  Blocks (Self-  │
│   Thermal)      │    │  (Linear Proj)  │    │  Attention)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   Output Heads  │
                        │   (Classification,│
                        │   Detection,     │
                        │   Segmentation)  │
                        └─────────────────┘
```

## Vision Transformer Architecture

### Patch Embedding

The key innovation in ViTs is converting images into sequences of patches:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Convolution for patch extraction (more efficient than manual reshaping)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Extract patches using convolution
        patches = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)

        # Reshape to sequence format
        patches = patches.flatten(2).transpose(1, 2)  # (batch_size, n_patches, embed_dim)

        return patches

class PatchEmbeddingWithPositional(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Positional embeddings for patches
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Dropout
        self.pos_drop = nn.Dropout(0.1)

    def forward(self, x):
        # Get patch embeddings
        patches = self.patch_embed(x)
        batch_size = patches.shape[0]

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1)

        # Add positional embeddings (class token gets position 0)
        pos_embed = torch.cat((torch.zeros(1, 1, self.pos_embed.size(-1)), self.pos_embed), dim=1)
        patches = patches + pos_embed
        patches = self.pos_drop(patches)

        return patches
```

### Transformer Blocks

The core of the Vision Transformer consists of transformer blocks:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Self-attention layer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # MLP layer
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        x = x + attn_out
        x = self.norm1(x)

        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)

        return x, attn_weights
```

### Complete Vision Transformer Implementation

```python
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbeddingWithPositional(
            img_size, patch_size, in_channels, embed_dim
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_dropout)
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding with positional encoding
        x = self.patch_embed(x)

        # Apply transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)

        # Take class token output
        x = self.norm(x[:, 0])

        # Classification
        x = self.head(x)

        return x, attention_weights
```

## Vision Transformer Variants

### DeiT (Data-efficient Image Transformers)

DeiT improves ViTs with distillation and training techniques:

```python
class DistilledVisionTransformer(nn.Module):
    def __init__(self, vit_model, teacher_model=None):
        super().__init__()
        self.vit = vit_model

        # Distillation token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, vit_model.patch_embed.embed_dim))

        # Distillation head
        self.dist_head = nn.Linear(vit_model.patch_embed.embed_dim, vit_model.head.out_features)

        # Initialize distillation token
        torch.nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x, teacher_output=None):
        batch_size = x.shape[0]

        # Patch embedding with class and distillation tokens
        patches = self.vit.patch_embed.patch_embed(x)
        cls_tokens = self.vit.patch_embed.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.dist_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, dist_tokens, patches), dim=1)

        # Add positional embeddings
        pos_embed = torch.cat((
            torch.zeros(1, 2, self.vit.patch_embed.embed_dim),
            self.vit.patch_embed.pos_embed
        ), dim=1)
        x = x + pos_embed
        x = self.vit.patch_embed.pos_drop(x)

        # Apply transformer blocks
        for block in self.vit.blocks:
            x, _ = block(x)

        # Apply final norm
        x = self.vit.norm(x)

        # Get outputs
        cls_output = self.vit.head(x[:, 0])
        dist_output = self.dist_head(x[:, 1])

        return cls_output, dist_output
```

### Swin Transformer (Shifted Windows)

Swin Transformers use hierarchical windows for efficiency:

```python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        ).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### Efficient Vision Transformers

For resource-constrained robotic systems:

```python
class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, transformer_dim, num_heads, mlp_ratio, patch_size=2):
        super().__init__()
        self.patch_size = patch_size

        # Local representation
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, transformer_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.SiLU()
        )

        # Global representation
        self.transformer = TransformerBlock(transformer_dim, num_heads, mlp_ratio)

        # Projection
        self.proj = nn.Sequential(
            nn.Conv2d(transformer_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Local representation
        local_features = self.local_rep(x)

        # Reshape for transformer
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        local_features = local_features.view(batch_size, -1, patch_h * patch_w).transpose(1, 2)

        # Apply transformer
        global_features, _ = self.transformer(local_features)

        # Reshape back
        global_features = global_features.transpose(1, 2).view(
            batch_size, -1, patch_h, patch_w
        )

        # Upsample to original size
        global_features = F.interpolate(
            global_features, size=(height, width), mode='bilinear', align_corners=False
        )

        # Apply projection and skip connection
        out = self.proj(global_features) + self.skip(x)

        return out
```

## Applications in Robotics

### Object Detection with Vision Transformers

```python
class ViTDet(nn.Module):
    def __init__(self, vit_backbone, num_classes=80):
        super().__init__()
        self.vit = vit_backbone

        # Feature pyramid network
        self.fpn = nn.ModuleList([
            nn.Conv2d(vit_backbone.embed_dim, 256, 1) for _ in range(5)  # P2-P6
        ])

        # Detection heads
        self.class_head = nn.Conv2d(256, num_classes, 3, padding=1)
        self.box_head = nn.Conv2d(256, 4, 3, padding=1)  # x, y, w, h

    def forward(self, x):
        # Forward through ViT backbone
        features, attention_weights = self.vit(x)

        # Reshape ViT features to spatial format
        batch_size, n_tokens, embed_dim = features.shape
        patch_size = int((n_tokens - 1) ** 0.5)  # Exclude class token
        spatial_features = features[:, 1:].reshape(batch_size, patch_size, patch_size, embed_dim)
        spatial_features = spatial_features.permute(0, 3, 1, 2)

        # Apply FPN
        fpn_features = []
        for i, fpn_layer in enumerate(self.fpn):
            # Upsample or downsample as needed
            feat = F.interpolate(spatial_features, scale_factor=2**i) if i < 2 else \
                   F.interpolate(spatial_features, scale_factor=2**(-i+2))
            fpn_features.append(fpn_layer(feat))

        # Apply detection heads
        class_outputs = [self.class_head(feat) for feat in fpn_features]
        box_outputs = [self.box_head(feat) for feat in fpn_features]

        return class_outputs, box_outputs
```

### Semantic Segmentation with Vision Transformers

```python
class SegmentationViT(nn.Module):
    def __init__(self, vit_backbone, num_classes=21):
        super().__init__()
        self.vit = vit_backbone

        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.Conv2d(vit_backbone.embed_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        # Upsampling to original resolution
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Forward through ViT
        features, attention_weights = self.vit(x)

        # Reshape features to spatial format
        batch_size, n_tokens, embed_dim = features.shape
        patch_size = int((n_tokens - 1) ** 0.5)
        spatial_features = features[:, 1:].reshape(batch_size, patch_size, patch_size, embed_dim)
        spatial_features = spatial_features.permute(0, 3, 1, 2)

        # Apply decoder
        seg_features = self.decoder(spatial_features)

        # Upsample to original image size
        segmentation = self.upsample(seg_features)

        return segmentation
```

### Robot Navigation and Scene Understanding

```python
class NavigationViT(nn.Module):
    def __init__(self, vit_backbone, action_space=4):
        super().__init__()
        self.vit = vit_backbone

        # Navigation-specific heads
        self.navigation_head = nn.Linear(vit_backbone.embed_dim, action_space)
        self.collision_head = nn.Linear(vit_backbone.embed_dim, 2)  # safe/unsafe
        self.goal_distance_head = nn.Linear(vit_backbone.embed_dim, 1)  # distance to goal

    def forward(self, image, goal_embedding=None):
        # Process image with ViT
        features, attention_weights = self.vit(image)

        # Use class token features for navigation
        cls_features = features[:, 0]  # Class token contains global information

        # Add goal information if provided
        if goal_embedding is not None:
            cls_features = torch.cat([cls_features, goal_embedding], dim=-1)

        # Generate navigation outputs
        actions = self.navigation_head(cls_features)
        collision_prob = torch.sigmoid(self.collision_head(cls_features))
        goal_distance = self.goal_distance_head(cls_features)

        return {
            'actions': actions,
            'collision_prob': collision_prob,
            'goal_distance': goal_distance,
            'attention_weights': attention_weights
        }
```

## Training Vision Transformers for Robotics

### Pre-training Strategies

#### Masked Autoencoders (MAE)

```python
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, vit_encoder, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = vit_encoder
        self.mask_ratio = mask_ratio

        # Decoder
        self.decoder_embed = nn.Linear(vit_encoder.patch_embed.embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, 8, 4.0, 0.1) for _ in range(8)
        ])

        # Prediction head
        self.prediction_head = nn.Linear(decoder_dim, 3 * 16 * 16)  # Patch size 16x16

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling tokens.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, ids_restore, ids_keep

    def forward(self, imgs):
        # Encode patches
        x = self.encoder.patch_embed(imgs)
        batch_size, n_patches, embed_dim = x.shape

        # Add positional embeddings
        x = x + self.encoder.patch_embed.pos_embed[:, 1:, :]  # Exclude class token

        # Random masking
        x_masked, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)

        # Add class token
        cls_token = self.encoder.patch_embed.cls_token.expand(batch_size, -1, -1)
        cls_tokens = torch.cat([cls_token, x_masked], dim=1)

        # Apply encoder blocks (only to visible patches)
        for block in self.encoder.blocks:
            cls_tokens, _ = block(cls_tokens)

        # Remove class token
        x = cls_tokens[:, 1:, :]

        # Apply decoder embed
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(batch_size, ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x, _ = block(x)

        # Predict pixels
        pred = self.prediction_head(x)

        return pred, ids_keep
```

### Fine-tuning for Robot Tasks

```python
def fine_tune_vit_for_robotics(pretrained_vit, robot_task_dataset, num_epochs=100):
    """
    Fine-tune a pre-trained ViT for a specific robotics task
    """
    # Add task-specific head
    task_head = nn.Linear(pretrained_vit.head.in_features, task_num_classes)

    # Create robot-specific model
    robot_vit = nn.Sequential([
        pretrained_vit,
        task_head
    ])

    # Only fine-tune the task head initially
    for param in pretrained_vit.parameters():
        param.requires_grad = False

    # Then fine-tune with lower learning rate
    optimizer = torch.optim.AdamW([
        {'params': task_head.parameters(), 'lr': 1e-3},
        {'params': pretrained_vit.parameters(), 'lr': 1e-5}  # Lower LR for pre-trained
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):
        for batch in robot_task_dataset:
            optimizer.zero_grad()

            # Forward pass
            outputs, attention_weights = robot_vit(batch['images'])
            loss = compute_robot_task_loss(outputs, batch['labels'])

            # Backward pass
            loss.backward()
            optimizer.step()

        scheduler.step()

    return robot_vit
```

## Attention Visualization and Interpretation

Understanding what Vision Transformers attend to:

```python
class ViTAttentionVisualizer:
    def __init__(self, vit_model):
        self.model = vit_model
        self.attention_maps = []

        # Register hooks to capture attention weights
        for i, block in enumerate(self.model.blocks):
            block.attn.register_forward_hook(
                lambda module, input, output, idx=i: self.save_attention(idx, output[1])
            )

    def save_attention(self, block_idx, attention_weights):
        self.attention_maps.append(attention_weights)

    def get_attention_map(self, image):
        """
        Get attention map for an image
        """
        self.attention_maps = []  # Clear previous attention maps

        with torch.no_grad():
            outputs, _ = self.model(image.unsqueeze(0))

        # Combine attention maps from all heads and layers
        combined_attention = torch.stack(self.attention_maps).mean(dim=0)  # Average across layers
        combined_attention = combined_attention.mean(dim=1)  # Average across heads

        # Remove class token attention and reshape
        spatial_attention = combined_attention[0, 1:, 1:]  # Remove class token

        return spatial_attention

    def visualize_attention(self, image, attention_map):
        """
        Visualize attention on the image
        """
        import matplotlib.pyplot as plt

        # Convert to numpy for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        attention_np = attention_map.cpu().numpy()

        # Reshape attention to match image dimensions
        patch_size = 16  # Assuming 16x16 patches
        img_size = image.shape[1]  # Assuming square image
        n_patches = img_size // patch_size
        attention_reshaped = attention_np.reshape(n_patches, n_patches)

        # Upsample attention to image size
        attention_upsampled = F.interpolate(
            torch.tensor(attention_reshaped).unsqueeze(0).unsqueeze(0),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(image_np)
        ax1.set_title('Input Image')
        ax1.axis('off')

        ax2.imshow(image_np)
        ax2.imshow(attention_upsampled, cmap='jet', alpha=0.5)
        ax2.set_title('Attention Overlay')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()
```

## Vision Transformers for Multi-Modal Robotics

### Vision-Language Integration

```python
class VisionLanguageViT(nn.Module):
    def __init__(self, vit_backbone, text_encoder, fusion_dim=768):
        super().__init__()
        self.vision_encoder = vit_backbone
        self.text_encoder = text_encoder

        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )

        # Fusion layers
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_proj = nn.Linear(fusion_dim * 2, fusion_dim)

    def forward(self, images, text_tokens):
        # Encode vision
        vision_features, vision_attn = self.vision_encoder(images)
        vision_cls = vision_features[:, 0]  # Class token

        # Encode text
        text_features = self.text_encoder(text_tokens)
        text_cls = text_features[:, 0]  # [CLS] token

        # Cross-attention fusion
        fused_features, cross_attn = self.cross_attention(
            vision_cls.unsqueeze(1),
            text_cls.unsqueeze(1),
            text_cls.unsqueeze(1)
        )

        # Combine features
        combined = torch.cat([vision_cls, fused_features.squeeze(1)], dim=-1)
        output = self.fusion_proj(combined)

        return output, cross_attn
```

### Multi-Scale Vision Transformers

```python
class MultiScaleViT(nn.Module):
    def __init__(self, base_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(224, 4, 3, base_dim)

        # Multi-scale transformer blocks
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.ModuleList([
                TransformerBlock(
                    base_dim * (2 ** i),
                    num_heads[i],
                    4.0,
                    0.1
                ) for _ in range(depths[i])
            ])
            self.stages.append(stage)

        # Feature aggregation
        self.feature_aggregator = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_dim * (2 ** (len(depths)-1)), 1000)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Process through stages
        for i, stage in enumerate(self.stages):
            for block in stage:
                x, _ = block(x)

        # Aggregate features
        x = self.feature_aggregator(x.transpose(1, 2)).squeeze(-1)
        x = self.classifier(x)

        return x
```

## Challenges and Limitations

### Computational Requirements

Vision Transformers can be computationally expensive:

#### Memory Optimization
```python
class MemoryEfficientViT(nn.Module):
    def __init__(self, original_vit):
        super().__init__()
        self.vit = original_vit

        # Apply gradient checkpointing for training
        self.apply_gradient_checkpointing()

    def apply_gradient_checkpointing(self):
        """
        Apply gradient checkpointing to save memory during training
        """
        def checkpoint_wrapper(module):
            def wrapper(*inputs):
                return torch.utils.checkpoint.checkpoint(module, *inputs)
            return wrapper

        for block in self.vit.blocks:
            block = torch.utils.checkpoint(self.checkpoint_block, block)

    def checkpoint_block(self, block, x):
        return block(x)

    def forward(self, x):
        # Forward pass with memory efficiency
        return self.vit(x)
```

### Data Requirements

Vision Transformers typically require large datasets for effective training:

#### Data Augmentation for ViTs
```python
import torchvision.transforms as T

def get_vit_augmentation():
    """
    Get appropriate data augmentation for ViT training
    """
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25)  # Helps with robustness
    ])
```

### Robotic-Specific Challenges

#### Real-time Performance
- **Model Compression**: Quantization and pruning for edge deployment
- **Efficient Architectures**: MobileViT, EdgeViT for resource-constrained robots
- **Hardware Acceleration**: Leveraging specialized AI chips

#### Safety and Reliability
- **Uncertainty Quantification**: Understanding model confidence
- **Robustness**: Handling distribution shifts in real environments
- **Interpretability**: Understanding model decisions for safety-critical applications

## Integration with Robot Systems

### ROS Integration Example

```python
# Example ROS node using Vision Transformer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class ViTRobotNode(Node):
    def __init__(self):
        super().__init__('vit_robot_node')

        # Load pre-trained ViT model
        self.vit_model = self.load_pretrained_vit()
        self.vit_model.eval()

        # ROS components
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Create publishers
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)

        self.get_logger().info('Vision Transformer Robot Node Initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Preprocess image for ViT
            processed_image = self.preprocess_image(cv_image)

            # Run inference
            with torch.no_grad():
                predictions, attention_weights = self.vit_model(processed_image)

                # Process predictions
                detections = self.process_predictions(predictions)

                # Publish detections
                detection_msg = String()
                detection_msg.data = str(detections)
                self.detection_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def preprocess_image(self, image):
        # Convert to tensor and normalize
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tensor_image = transform(image).unsqueeze(0)
        return tensor_image

    def process_predictions(self, predictions):
        # Convert model outputs to meaningful detections
        probabilities = torch.softmax(predictions, dim=-1)
        top_classes = torch.topk(probabilities, k=5, dim=-1)

        detections = []
        for i in range(top_classes.values.shape[0]):
            for j in range(5):
                class_id = top_classes.indices[i, j].item()
                confidence = top_classes.values[i, j].item()
                detections.append({
                    'class_id': class_id,
                    'confidence': confidence
                })

        return detections
```

## Future Directions

### Emerging Architectures

#### Convolutional Vision Transformers (CvT)
- **Hybrid Approach**: Combining convolutions with transformers
- **Improved Inductive Bias**: Better spatial locality than pure ViTs
- **Efficiency**: More parameter-efficient than standard ViTs

#### Swin Transformer V2 and Beyond
- **Scalability**: Training larger models more efficiently
- **Transfer Learning**: Better zero-shot and few-shot capabilities
- **Multi-modal**: Integration with language and other modalities

### Robotics-Specific Innovations

#### Embodied Vision Transformers
- **3D Awareness**: Understanding 3D structure from 2D images
- **Temporal Reasoning**: Processing video sequences for dynamic tasks
- **Sensor Fusion**: Combining multiple sensor modalities

#### Continual Learning
- **Catastrophic Forgetting**: Preventing loss of previously learned skills
- **Life-long Learning**: Continuously learning new tasks
- **Adaptation**: Adapting to new environments and conditions

## Learning Summary

Vision Transformers have transformed computer vision in robotics by providing:

- **Global Attention** mechanisms that capture long-range dependencies
- **Scalability** with large datasets and model sizes
- **Multi-modal Integration** capabilities for vision-language tasks
- **Transfer Learning** effectiveness from pre-trained models
- **Interpretability** through attention visualization
- **Flexible Architectures** adaptable to various robotic tasks

Vision Transformers continue to evolve with new variants and applications emerging regularly in robotics.

## Exercises

1. Implement a basic Vision Transformer from scratch and train it on a small robotics-related dataset (e.g., object recognition in a robotic environment). Compare its performance with a CNN-based approach.

2. Design a Vision Transformer architecture specifically for robotic navigation that can process both RGB and depth images simultaneously. Include attention visualization capabilities.

3. Research and implement an efficient Vision Transformer variant (like MobileViT or EdgeViT) suitable for deployment on a resource-constrained robotic platform. Analyze the trade-offs between accuracy and computational efficiency.
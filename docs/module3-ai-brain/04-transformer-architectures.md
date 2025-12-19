---
sidebar_position: 4
title: "Transformer Architectures"
---

# Transformer Architectures

## Learning Outcomes
By the end of this chapter, you will be able to:
- Understand the fundamental architecture of transformer models
- Implement attention mechanisms for robotic applications
- Apply transformer models to various robotics tasks
- Evaluate the advantages and limitations of transformers in robotics
- Design transformer-based systems for multi-modal robotics applications

## Introduction to Transformer Architectures

Transformers have revolutionized artificial intelligence by introducing the attention mechanism, which allows models to focus on the most relevant parts of input data. Originally designed for natural language processing, transformers have expanded to computer vision, robotics, and multi-modal applications. In robotics, transformers enable more sophisticated understanding of complex environments and facilitate better decision-making by attending to relevant sensory information.

### The Attention Revolution

Traditional neural networks process information sequentially or with fixed patterns, but transformers can dynamically focus on different parts of the input based on relevance:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for queries, keys, values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back to original dimensions
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(attention_output)

        return output, attention_weights
```

### Why Transformers for Robotics?

Transformers offer several advantages for robotics applications:

#### Dynamic Attention
- **Selective Focus**: Robots can focus on relevant objects or regions
- **Context Awareness**: Understanding relationships between different elements
- **Adaptive Processing**: Adjusting attention based on task requirements

#### Long-Range Dependencies
- **Temporal Reasoning**: Understanding sequences of events
- **Spatial Relationships**: Capturing relationships across large environments
- **Multi-step Planning**: Reasoning about long-term goals

```
Transformer Architecture for Robotics
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Input  │───→│ Transformer     │───→│   Robot Output  │
│   (Vision,      │    │   Encoder-      │    │   (Actions,     │
│   Language,     │    │   Decoder       │    │   Plans,        │
│   Tactile)      │    │   (Attention)   │    │   Decisions)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   Cross-Modal   │
                        │   Integration   │
                        └─────────────────┘
```

## The Transformer Architecture

### Encoder Architecture

The transformer encoder processes input sequences and generates contextual representations:

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attended, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attended))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attention_weights

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)

        return x, attention_weights
```

### Decoder Architecture

The decoder generates output sequences, typically used in sequence-to-sequence tasks:

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        # Masked self-attention
        self.masked_attention = MultiHeadAttention(d_model, num_heads)

        # Cross-attention with encoder
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attended, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attended))

        # Cross-attention with encoder
        attended, attention_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attended))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, attention_weights

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, encoder_output, src_mask, tgt_mask)
            attention_weights.append(attn_weights)

        return x, attention_weights
```

### Positional Encoding

Since transformers don't have inherent sequential order, positional encoding is crucial:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Apply sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]
```

## Attention Mechanisms in Robotics

### Self-Attention for Scene Understanding

Self-attention allows robots to understand relationships within their environment:

```python
class SceneUnderstandingTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6):
        super(SceneUnderstandingTransformer, self).__init__()

        # Visual feature encoder
        self.visual_encoder = nn.Conv2d(3, d_model, kernel_size=16, stride=16)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        self.transformer = TransformerEncoder(num_layers, d_model, num_heads, d_model * 4)

        # Output heads
        self.object_detection_head = nn.Linear(d_model, 80)  # COCO classes
        self.instance_segmentation_head = nn.Linear(d_model, 256)  # Embedding
        self.pose_estimation_head = nn.Linear(d_model, 7)  # x, y, z, qw, qx, qy, qz

    def forward(self, image):
        # Extract visual features
        features = self.visual_encoder(image)
        batch_size, channels, height, width = features.shape

        # Reshape to sequence format
        features = features.view(batch_size, channels, -1).transpose(1, 2)

        # Add positional encoding
        features = self.pos_encoding(features)

        # Apply transformer
        attended_features, attention_weights = self.transformer(features)

        # Apply output heads
        object_logits = self.object_detection_head(attended_features)
        segmentation_embeddings = self.instance_segmentation_head(attended_features)
        pose_predictions = self.pose_estimation_head(attended_features)

        return {
            'objects': object_logits,
            'segmentation': segmentation_embeddings,
            'poses': pose_predictions,
            'attention_weights': attention_weights
        }
```

### Cross-Attention for Multi-Modal Integration

Cross-attention enables integration of different sensory modalities:

```python
class MultiModalTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(MultiModalTransformer, self).__init__()

        # Modality-specific encoders
        self.visual_encoder = nn.Linear(2048, d_model)  # ResNet features
        self.language_encoder = nn.Linear(768, d_model)  # BERT features
        self.tactile_encoder = nn.Linear(128, d_model)   # Tactile features

        # Cross-attention modules
        self.vision_language_cross_attn = MultiHeadAttention(d_model, num_heads)
        self.vision_tactile_cross_attn = MultiHeadAttention(d_model, num_heads)
        self.language_tactile_cross_attn = MultiHeadAttention(d_model, num_heads)

        # Fusion layers
        self.fusion_network = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, visual_features, language_features, tactile_features):
        # Encode modalities
        visual_encoded = self.visual_encoder(visual_features)
        language_encoded = self.language_encoder(language_features)
        tactile_encoded = self.tactile_encoder(tactile_features)

        # Cross-attention between modalities
        # Vision-language interaction
        vision_lang, _ = self.vision_language_cross_attn(
            visual_encoded, language_encoded, language_encoded
        )

        # Vision-tactile interaction
        vision_tactile, _ = self.vision_tactile_cross_attn(
            visual_encoded, tactile_encoded, tactile_encoded
        )

        # Language-tactile interaction
        lang_tactile, _ = self.language_tactile_cross_attn(
            language_encoded, tactile_encoded, tactile_encoded
        )

        # Concatenate and fuse
        combined = torch.cat([vision_lang, vision_tactile, lang_tactile], dim=-1)
        fused_features = self.fusion_network(combined)

        return fused_features
```

### Attention Visualization and Interpretability

Understanding what the robot is attending to:

```python
class AttentionVisualizer:
    def __init__(self):
        pass

    def visualize_attention(self, attention_weights, input_tokens, output_tokens=None):
        """
        Visualize attention weights between input and output tokens
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]

        # Plot attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attention[0].cpu().detach().numpy(),
            xticklabels=input_tokens,
            yticklabels=output_tokens or input_tokens,
            cmap='viridis'
        )
        plt.title('Attention Heatmap')
        plt.show()

    def robot_attention_map(self, image, attention_weights, patch_size=16):
        """
        Create attention visualization for robot vision
        """
        # Reshape attention weights to image dimensions
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        height = width = int(seq_len ** 0.5)

        # Average across heads
        avg_attention = attention_weights.mean(dim=1).squeeze(0)

        # Reshape to image format
        attention_map = avg_attention.view(height, width)

        # Upsample to original image size
        attention_map = F.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=(image.shape[-2], image.shape[-1]),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        return attention_map
```

## Transformer Applications in Robotics

### Natural Language Understanding for Robots

Transformers enable robots to understand and respond to natural language commands:

```python
class RobotLanguageTransformer(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, num_heads=12, num_layers=12):
        super(RobotLanguageTransformer, self).__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer encoder
        self.transformer = TransformerEncoder(num_layers, d_model, num_heads, d_model * 4)

        # Robot-specific output heads
        self.action_head = nn.Linear(d_model, 20)  # 20 possible robot actions
        self.object_head = nn.Linear(d_model, 100)  # 100 possible objects
        self.location_head = nn.Linear(d_model, 50)  # 50 possible locations
        self.attribute_head = nn.Linear(d_model, 30)  # 30 possible attributes

    def forward(self, input_ids, attention_mask=None):
        # Embed tokens
        embeddings = self.token_embedding(input_ids)

        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)

        # Apply transformer
        attended_features, attention_weights = self.transformer(embeddings, attention_mask)

        # Apply output heads
        actions = self.action_head(attended_features)
        objects = self.object_head(attended_features)
        locations = self.location_head(attended_features)
        attributes = self.attribute_head(attended_features)

        return {
            'actions': actions,
            'objects': objects,
            'locations': locations,
            'attributes': attributes,
            'attention_weights': attention_weights
        }

    def parse_command(self, command_text):
        """
        Parse natural language command and extract structured information
        """
        # Tokenize command
        tokens = self.tokenize(command_text)
        input_ids = torch.tensor([tokens])

        # Forward pass
        outputs = self.forward(input_ids)

        # Extract most likely action, object, location
        action = torch.argmax(outputs['actions'], dim=-1)
        obj = torch.argmax(outputs['objects'], dim=-1)
        location = torch.argmax(outputs['locations'], dim=-1)

        return {
            'action': action,
            'object': obj,
            'location': location,
            'confidence': torch.softmax(outputs['actions'], dim=-1).max(dim=-1)
        }
```

### Task Planning with Transformers

Transformers can be used for high-level task planning and reasoning:

```python
class TaskPlanningTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6):
        super(TaskPlanningTransformer, self).__init__()

        # Environment state encoder
        self.state_encoder = nn.Linear(128, d_model)  # Encode environment state

        # Task goal encoder
        self.goal_encoder = nn.Linear(64, d_model)    # Encode goal state

        # Action space encoder
        self.action_encoder = nn.Linear(32, d_model)  # Encode possible actions

        # Transformer for planning
        self.transformer = TransformerEncoder(num_layers, d_model, num_heads, d_model * 4)

        # Output heads
        self.action_sequence_head = nn.Linear(d_model, 100)  # Predict action sequence
        self.value_head = nn.Linear(d_model, 1)              # Estimate value of state

    def forward(self, current_state, goal_state, possible_actions):
        # Encode inputs
        state_emb = self.state_encoder(current_state).unsqueeze(1)
        goal_emb = self.goal_encoder(goal_state).unsqueeze(1)
        action_embs = self.action_encoder(possible_actions)

        # Concatenate all embeddings
        all_embs = torch.cat([state_emb, goal_emb, action_embs], dim=1)

        # Apply transformer
        attended, attention_weights = self.transformer(all_embs)

        # Extract outputs
        state_features = attended[:, 0, :]  # State embedding
        action_features = attended[:, 2:, :]  # Action embeddings

        # Predict action sequence
        action_sequence = self.action_sequence_head(state_features)

        # Estimate state value
        state_value = self.value_head(state_features)

        return {
            'action_sequence': action_sequence,
            'state_value': state_value,
            'attention_weights': attention_weights
        }

    def plan_task(self, initial_state, goal_state):
        """
        Generate a sequence of actions to achieve the goal
        """
        # Get possible actions for current state
        possible_actions = self.get_possible_actions(initial_state)

        # Plan using transformer
        plan_outputs = self.forward(initial_state, goal_state, possible_actions)

        # Decode action sequence
        action_sequence = torch.argmax(plan_outputs['action_sequence'], dim=-1)

        return action_sequence
```

### Multi-Step Decision Making

Transformers excel at multi-step reasoning and planning:

```python
class SequentialDecisionTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, max_steps=10):
        super(SequentialDecisionTransformer, self).__init__()

        self.max_steps = max_steps
        self.d_model = d_model

        # State encoder
        self.state_encoder = nn.Linear(256, d_model)

        # Step encoder (for temporal reasoning)
        self.step_encoder = nn.Embedding(max_steps, d_model)

        # Transformer for sequential decision making
        self.transformer = TransformerEncoder(6, d_model, num_heads, d_model * 4)

        # Output head for action prediction
        self.action_head = nn.Linear(d_model, 50)  # 50 possible actions
        self.continuation_head = nn.Linear(d_model, 2)  # Continue or stop

    def forward(self, states_sequence, step_numbers):
        # Encode states
        state_embs = self.state_encoder(states_sequence)

        # Add step information
        step_embs = self.step_encoder(step_numbers)
        combined_embs = state_embs + step_embs

        # Apply transformer
        attended, attention_weights = self.transformer(combined_embs)

        # Predict actions and continuation
        actions = self.action_head(attended)
        continuation = torch.sigmoid(self.continuation_head(attended))

        return {
            'actions': actions,
            'continuation': continuation,
            'attention_weights': attention_weights
        }

    def execute_plan(self, initial_state, max_steps=10):
        """
        Execute a multi-step plan using the transformer
        """
        states = [initial_state]
        actions = []
        step = 0

        while step < max_steps:
            # Prepare input for current step
            current_state = states[-1].unsqueeze(0)
            step_tensor = torch.tensor([step]).unsqueeze(0)

            # Get action prediction
            outputs = self.forward(current_state, step_tensor)

            # Select action
            action = torch.argmax(outputs['actions'], dim=-1)
            actions.append(action)

            # Check if should continue
            should_continue = outputs['continuation'][:, :, 1] > 0.5

            if not should_continue:
                break

            # Simulate next state (in real application, this would be real environment)
            next_state = self.simulate_next_state(current_state, action)
            states.append(next_state)

            step += 1

        return actions, states
```

## Vision-Language Transformers for Robotics

### CLIP-based Robot Perception

```python
class VisionLanguageRobot(nn.Module):
    def __init__(self, vision_model, text_model, d_model=512):
        super(VisionLanguageRobot, self).__init__()

        self.vision_model = vision_model  # e.g., VisionTransformer
        self.text_model = text_model      # e.g., BERT

        # Projection layers to common space
        self.vision_projection = nn.Linear(768, d_model)
        self.text_projection = nn.Linear(768, d_model)

        # Temperature parameter for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def encode_images(self, images):
        # Extract visual features
        vision_features = self.vision_model(images)
        vision_features = self.vision_projection(vision_features)
        # Normalize features
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        return vision_features

    def encode_texts(self, texts):
        # Extract text features
        text_features = self.text_model(texts)
        text_features = self.text_projection(text_features)
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, texts):
        # Encode both modalities
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)

        # Compute similarity scores
        logits_per_image = torch.matmul(image_features, text_features.t()) * self.logit_scale
        logits_per_text = logits_per_image.t()

        return {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_features': image_features,
            'text_features': text_features
        }

    def find_object_by_description(self, image, object_descriptions):
        """
        Find objects in image based on text descriptions
        """
        image_features = self.encode_images(image.unsqueeze(0))
        text_features = self.encode_texts(object_descriptions)

        # Compute similarities
        similarities = torch.matmul(image_features, text_features.t())

        # Find best matches
        best_matches = torch.argmax(similarities, dim=-1)

        return best_matches, similarities
```

### Flamingo-style Architectures

```python
class FlamingoRobot(nn.Module):
    def __init__(self, vision_encoder, language_model, d_model=2048):
        super(FlamingoRobot, self).__init__()

        # Vision encoder (frozen)
        self.vision_encoder = vision_encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Language model (frozen)
        self.language_model = language_model
        for param in self.language_model.parameters():
            param.requires_grad = False

        # Cross-attention layers for vision-language fusion
        self.vision_language_fusion = nn.ModuleList([
            MultiHeadAttention(d_model, 8) for _ in range(4)
        ])

        # Projection layers
        self.vision_projection = nn.Linear(1024, d_model)
        self.language_projection = nn.Linear(4096, d_model)

    def forward(self, images, text_inputs, attention_mask=None):
        # Encode visual information
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_projection(vision_features)

        # Encode text information
        text_features = self.language_model(text_inputs, attention_mask=attention_mask)
        text_features = self.language_projection(text_features)

        # Fuse vision and language through cross-attention
        fused_features = text_features
        for fusion_layer in self.vision_language_fusion:
            fused_features, _ = fusion_layer(
                fused_features, vision_features, vision_features
            )

        return fused_features
```

## Training Transformers for Robotics

### Pre-training Strategies

#### Masked Language Modeling for Robot Commands
```python
class RobotMLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6):
        super(RobotMLM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = TransformerEncoder(num_layers, d_model, num_heads, d_model * 4)
        self.mask_token_id = vocab_size - 1  # Last token is mask token
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embed tokens
        embeddings = self.embedding(input_ids)
        pos_embeddings = PositionalEncoding(d_model)(embeddings)

        # Apply transformer
        attended, _ = self.transformer(pos_embeddings, attention_mask)

        # Project to vocabulary
        logits = self.output_projection(attended)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {
            'logits': logits,
            'loss': loss
        }

    def mask_tokens(self, input_ids, mask_ratio=0.15):
        """
        Randomly mask tokens for MLM training
        """
        batch_size, seq_len = input_ids.shape
        mask = torch.rand(batch_size, seq_len) < mask_ratio

        # Create labels (copy of input, -100 for masked tokens)
        labels = input_ids.clone()
        labels[~mask] = -100  # Ignore non-masked tokens in loss

        # Replace masked tokens with mask token
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.mask_token_id

        return masked_input_ids, labels
```

### Fine-tuning for Specific Tasks

```python
def fine_tune_robot_transformer(pretrained_model, task_dataset, num_epochs=10):
    """
    Fine-tune a pre-trained transformer for a specific robotics task
    """
    # Add task-specific head
    task_head = nn.Linear(pretrained_model.config.d_model, task_num_classes)

    # Combine models
    model = nn.Sequential([
        pretrained_model,
        task_head
    ])

    # Only fine-tune task head initially
    for param in pretrained_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': task_head.parameters(), 'lr': 1e-3},
        {'params': pretrained_model.parameters(), 'lr': 1e-5}  # Smaller LR for pre-trained
    ])

    for epoch in range(num_epochs):
        for batch in task_dataset:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch['input_ids'])
            loss = compute_task_loss(outputs, batch['labels'])

            # Backward pass
            loss.backward()
            optimizer.step()

    return model
```

## Challenges and Limitations

### Computational Requirements

Transformers are computationally intensive, which presents challenges for robotics:

#### Memory Optimization
- **Gradient Checkpointing**: Reducing memory usage during training
- **Mixed Precision Training**: Using 16-bit instead of 32-bit precision
- **Model Parallelism**: Distributing model across multiple GPUs

#### Inference Optimization
```python
class OptimizedRobotTransformer(nn.Module):
    def __init__(self, original_model):
        super(OptimizedRobotTransformer, self).__init__()

        # Quantize the model
        self.quantized_model = torch.quantization.quantize_dynamic(
            original_model, {nn.Linear}, dtype=torch.qint8
        )

        # Prune unnecessary connections
        self.pruned_model = self.prune_model(self.quantized_model)

    def forward(self, x):
        return self.pruned_model(x)

    def prune_model(self, model):
        # Example pruning implementation
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune 20% of weights
                prune.l1_unstructured(module, name='weight', amount=0.2)
        return model
```

### Real-time Constraints

#### Efficient Attention Mechanisms
- **Sparse Attention**: Computing attention only for relevant positions
- **Linear Attention**: Approximating attention with linear complexity
- **Memory-Efficient Attention**: Using techniques like FlashAttention

#### Model Compression
- **Knowledge Distillation**: Training smaller "student" models
- **Parameter Sharing**: Sharing parameters across layers
- **Efficient Architectures**: Using more efficient transformer variants

### Safety and Reliability

#### Uncertainty Quantification
```python
class UncertaintyAwareTransformer(nn.Module):
    def __init__(self, base_model, num_samples=10):
        super(UncertaintyAwareTransformer, self).__init__()
        self.base_model = base_model
        self.num_samples = num_samples

    def forward_with_uncertainty(self, x):
        # Run model multiple times with dropout
        outputs = []
        for _ in range(self.num_samples):
            output = self.base_model(x)
            outputs.append(output)

        # Compute mean and uncertainty
        outputs = torch.stack(outputs, dim=0)
        mean_output = outputs.mean(dim=0)
        uncertainty = outputs.var(dim=0)

        return mean_output, uncertainty
```

## Integration with Robot Systems

### ROS Integration

```python
# Example ROS node using transformer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

class TransformerRobotNode(Node):
    def __init__(self):
        super().__init__('transformer_robot_node')

        # Load pre-trained transformer model
        self.transformer_model = self.load_transformer_model()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.command_sub = self.create_subscription(
            String, '/robot_command', self.command_callback, 10
        )

        # Create publishers
        self.action_pub = self.create_publisher(String, '/robot_action', 10)

        self.get_logger().info('Transformer Robot Node Initialized')

    def image_callback(self, msg):
        # Convert ROS image to tensor
        image_tensor = self.ros_image_to_tensor(msg)

        # Process with transformer
        with torch.no_grad():
            features = self.transformer_model.encode_images(image_tensor)

        # Store features for later use
        self.current_features = features

    def command_callback(self, msg):
        # Process command with transformer
        command_features = self.transformer_model.encode_texts([msg.data])

        # Generate action based on current scene and command
        action = self.generate_action(self.current_features, command_features)

        # Publish action
        action_msg = String()
        action_msg.data = action
        self.action_pub.publish(action_msg)
```

## Future Directions

### Efficient Transformers

#### Mobile-Optimized Architectures
- **MobileViT**: Efficient vision transformers for mobile devices
- **Efficient Attention**: Linear-complexity attention mechanisms
- **Hardware-Aware Transformers**: Models optimized for specific hardware

#### Edge Deployment
- **ONNX Conversion**: Converting transformers for edge deployment
- **TensorRT Optimization**: Optimizing for NVIDIA GPUs
- **Embedded Optimization**: Optimizing for robotics hardware

### Multi-Modal Transformers

#### Unified Architectures
- **Perceiver**: General-purpose architecture for multiple modalities
- **Flamingo**: Vision-language models for instruction following
- **PaLM-E**: Embodied multimodal language models

#### Continuous Learning
- **Continual Transformers**: Models that learn continuously
- **Meta-Learning**: Learning to learn new tasks quickly
- **Few-Shot Learning**: Learning from few examples

## Learning Summary

Transformer architectures have transformed robotics by providing:

- **Attention Mechanisms** that allow robots to focus on relevant information
- **Multi-Modal Integration** capabilities for combining different sensory inputs
- **Sequential Reasoning** for multi-step task planning
- **Natural Language Understanding** for human-robot interaction
- **Scalable Architectures** that can handle complex robotic tasks
- **Transfer Learning** capabilities across different robotic applications

Transformers continue to evolve with new architectures and applications emerging regularly in robotics.

## Exercises

1. Implement a simple transformer model for a robotics task such as object detection or action prediction. Train it on a small dataset and evaluate its performance.

2. Design a multi-modal transformer that combines visual and language inputs for a robot instruction following task. Include attention visualization capabilities.

3. Research and compare different efficient transformer architectures (MobileViT, Swin Transformer, etc.) for robotics applications. Analyze their trade-offs in terms of accuracy, speed, and memory usage.
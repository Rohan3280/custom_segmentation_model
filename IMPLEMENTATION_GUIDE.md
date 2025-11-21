# Implementation Guide: DINOv2 ViT-L/14 with 25 Layers

This guide provides step-by-step instructions for implementing the DINOv2 ViT-L/14 model with 25 layers for glacial lake segmentation.

---

## Overview

The goal is to implement:
1. DINOv2 ViT-L/14 backbone (24 blocks)
2. 25th transformer block (identity initialized)
3. Segmentation decoder (U-Net style)
4. Segmentation head (1×1 conv)
5. Complete training and inference pipeline

---

## Step 1: Install Dependencies

```bash
pip install torch torchvision
pip install timm  # For DINOv2 models
pip install segmentation-models-pytorch  # Optional, for segmentation utilities
pip install einops  # For tensor operations
pip install opencv-python pillow numpy
pip install pyyaml tqdm
```

---

## Step 2: Implement DINOv2 Backbone with 25th Block

### 2.1 Load DINOv2 Pretrained Weights

```python
import torch
import torch.nn as nn
from timm import create_model

def load_dinov2_pretrained():
    """Load DINOv2 ViT-L/14 pretrained model"""
    model = create_model(
        'vit_large_patch14_dinov2',
        pretrained=True,
        num_classes=0,  # Remove classification head
        img_size=518,
    )
    return model
```

### 2.2 Extract Encoder (24 Blocks)

```python
def extract_encoder(model):
    """Extract encoder from DINOv2 model"""
    # DINOv2 ViT-L/14 has 24 transformer blocks
    encoder = model.blocks  # 24 blocks
    return encoder
```

### 2.3 Create 25th Transformer Block

```python
class TransformerBlock(nn.Module):
    """DINOv2 Transformer Block with LayerScale"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, init_values=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, qkv_bias=qkv_bias, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
    
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.gamma_2 is not None:
            x = self.gamma_2 * x
        x = residual + x
        
        return x

def create_25th_block(dim=1024, num_heads=16, identity_init=True):
    """Create 25th transformer block with identity initialization"""
    if identity_init:
        # Identity initialization: zero weights, zero LayerScale gamma
        block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            init_values=0.0  # Zero LayerScale gamma
        )
        # Initialize weights to zero
        nn.init.zeros_(block.attn.in_proj_weight)
        nn.init.zeros_(block.attn.out_proj.weight)
        nn.init.zeros_(block.mlp[0].weight)
        nn.init.zeros_(block.mlp[2].weight)
    else:
        # Standard initialization
        block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            init_values=1e-5  # Small LayerScale gamma
        )
    return block
```

### 2.4 Create Complete Encoder (25 Blocks)

```python
class DINOv2Encoder25Blocks(nn.Module):
    """DINOv2 ViT-L/14 encoder with 25 blocks"""
    def __init__(self, pretrained=True, identity_init_25th=True):
        super().__init__()
        # Load DINOv2 pretrained model
        dinov2 = load_dinov2_pretrained()
        
        # Extract patch embedding
        self.patch_embed = dinov2.patch_embed
        self.cls_token = dinov2.cls_token
        self.pos_embed = dinov2.pos_embed
        
        # Extract 24 transformer blocks
        self.blocks = nn.ModuleList(list(dinov2.blocks))
        
        # Add 25th transformer block
        self.block_25 = create_25th_block(
            dim=1024,
            num_heads=16,
            identity_init=identity_init_25th
        )
        
        # Norm layer
        self.norm = dinov2.norm
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # 24 pretrained blocks
        for block in self.blocks:
            x = block(x)
        
        # 25th block
        x = self.block_25(x)
        
        # Norm
        x = self.norm(x)
        
        return x
```

---

## Step 3: Implement Segmentation Decoder

### 3.1 U-Net Style Decoder

```python
class SegmentationDecoder(nn.Module):
    """U-Net style decoder for segmentation"""
    def __init__(self, encoder_dim=1024, decoder_dims=[512, 256, 128, 64], num_classes=1):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        
        # Initial projection
        self.initial_proj = nn.Conv2d(encoder_dim, decoder_dims[0], 1)
        
        # Decoder blocks
        for i in range(len(decoder_dims) - 1):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_dims[i],
                        decoder_dims[i + 1],
                        kernel_size=2,
                        stride=2
                    ),
                    nn.BatchNorm2d(decoder_dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        decoder_dims[i + 1],
                        decoder_dims[i + 1],
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(decoder_dims[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(
            decoder_dims[-1],
            num_classes,
            kernel_size=1
        )
    
    def forward(self, x, image_size=(518, 518)):
        # x shape: (batch_size, num_patches + 1, encoder_dim)
        # Remove CLS token
        x = x[:, 1:]  # Remove CLS token
        
        # Reshape to spatial dimensions
        # Assuming 518x518 input with 14x14 patches: 37x37 patches
        patch_size = 14
        h = w = int((x.shape[1]) ** 0.5)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)
        
        # Initial projection
        x = self.initial_proj(x)
        
        # Decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        # Upsample to original image size
        x = nn.functional.interpolate(
            x,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        return x
```

---

## Step 4: Create Complete Model

### 4.1 Complete Segmentation Model

```python
class DINOv2SegmentationModel(nn.Module):
    """Complete DINOv2 segmentation model with 25 blocks"""
    def __init__(
        self,
        pretrained=True,
        identity_init_25th=True,
        num_classes=1,
        image_size=518
    ):
        super().__init__()
        self.image_size = image_size
        
        # Encoder (25 blocks)
        self.encoder = DINOv2Encoder25Blocks(
            pretrained=pretrained,
            identity_init_25th=identity_init_25th
        )
        
        # Decoder
        self.decoder = SegmentationDecoder(
            encoder_dim=1024,
            decoder_dims=[512, 256, 128, 64],
            num_classes=num_classes
        )
    
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.decoder(x, image_size=(self.image_size, self.image_size))
        
        return x
```

---

## Step 5: Update Data Pipeline

### 5.1 Segmentation Dataset

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class SegmentationDataset(Dataset):
    """Dataset for segmentation with masks"""
    def __init__(self, image_dir, mask_dir, transform=None, image_size=518):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get list of images
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.tif'))
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.image_size, self.image_size))
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            # Create dummy mask if not available
            mask = torch.zeros((1, self.image_size, self.image_size))
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
```

### 5.2 Data Augmentation

```python
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

class SegmentationTransform:
    """Data augmentation for segmentation"""
    def __init__(self, image_size=518):
        self.image_size = image_size
    
    def __call__(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)
        
        # Random rotation
        angle = random.uniform(-10, 10)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        
        # Normalize image
        image = F.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        return image, mask
```

---

## Step 6: Implement Loss Functions

### 6.1 Lovasz-Dice-BCE Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LovaszDiceBCELoss(nn.Module):
    """Combined Lovasz-Dice-BCE loss for segmentation"""
    def __init__(self, lovasz_weight=0.3, dice_weight=0.3, bce_weight=0.4):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        """Dice loss"""
        pred = torch.sigmoid(pred)
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def lovasz_loss(self, pred, target):
        """Lovasz loss (simplified version)"""
        # Simplified Lovasz loss
        # For full implementation, see: https://github.com/bermanmaxim/LovaszSoftmax
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = intersection / (union + 1e-8)
        return 1 - iou
    
    def forward(self, pred, target):
        # BCE loss
        bce = self.bce_loss(pred, target)
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # Lovasz loss
        lovasz = self.lovasz_loss(pred, target)
        
        # Combined loss
        loss = (
            self.lovasz_weight * lovasz +
            self.dice_weight * dice +
            self.bce_weight * bce
        )
        
        return loss
```

---

## Step 7: Implement Metrics

### 7.1 IoU Metric

```python
import torch

def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU (Intersection over Union)"""
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    iou = intersection / (union + 1e-8)
    return iou.item()

def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice score"""
    pred = torch.sigmoid(pred) > threshold
    target = target > threshold
    
    intersection = (pred & target).float().sum()
    dice = (2.0 * intersection) / (pred.float().sum() + target.float().sum() + 1e-8)
    return dice.item()
```

---

## Step 8: Update Training Loop

### 8.1 Training Function

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-4,
    device='cuda'
):
    """Train segmentation model"""
    model = model.to(device)
    criterion = LovaszDiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train IoU: {train_iou/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val IoU: {val_iou/len(val_loader):.4f}')
        
        # Save best model
        if val_iou / len(val_loader) > best_iou:
            best_iou = val_iou / len(val_loader)
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'  Saved best model with IoU: {best_iou:.4f}')
    
    return model
```

---

## Step 9: Update Model Files

### 9.1 Update `src/models/vit_backbone.py`

Replace the existing implementation with the DINOv2 implementation from Step 2.

### 9.2 Update `src/models/decoder_unet.py`

Add the decoder implementation from Step 3.

### 9.3 Update `src/models/heads.py`

Add the segmentation head implementation from Step 3.

### 9.4 Update `src/losses/lovasz_dice_bce.py`

Add the loss function implementation from Step 6.

### 9.5 Update `src/metrics/iou_pixelwise.py`

Add the metrics implementation from Step 7.

### 9.6 Update `src/dataset/dataloader.py`

Replace with the segmentation dataset from Step 5.

---

## Step 10: Update Configuration

### 10.1 Update `configs/default.yaml`

```yaml
model:
  type: dinov2_vit_l_14
  num_layers: 25
  patch_size: 14
  hidden_dim: 1024
  num_heads: 16
  image_size: 518
  num_classes: 1
  identity_init_25th: true

train:
  batch_size: 8
  epochs: 50
  learning_rate: 1e-4
  optimizer: adamw
  scheduler: cosine

data:
  dataset: sentinel2
  train_dir: data/processed/train
  val_dir: data/processed/val
  test_dir: data/processed/test
  image_size: 518
  num_channels: 3

loss:
  type: lovasz_dice_bce
  lovasz_weight: 0.3
  dice_weight: 0.3
  bce_weight: 0.4

metrics:
  - iou
  - dice
  - pixel_accuracy
```

---

## Testing

### Test Model Forward Pass

```python
import torch
from src.models.vit_backbone import DINOv2SegmentationModel

# Create model
model = DINOv2SegmentationModel(
    pretrained=True,
    identity_init_25th=True,
    num_classes=1,
    image_size=518
)

# Test forward pass
x = torch.randn(2, 3, 518, 518)
output = model(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {output.shape}')  # Should be (2, 1, 518, 518)
```

---

## Next Steps

1. **Implement the code**: Follow the steps above to implement the DINOv2 model with 25 layers
2. **Test with sample data**: Test the model with sample images and masks
3. **Train on full dataset**: Train the model on the full GLOF dataset
4. **Evaluate performance**: Evaluate the model using IoU, Dice score, and pixel accuracy
5. **Compare with original DINOv2**: Compare performance with the original 24-layer DINOv2 model

---

*For detailed analysis, see `CODEBASE_ANALYSIS.md` and `ANALYSIS_SUMMARY.md`*





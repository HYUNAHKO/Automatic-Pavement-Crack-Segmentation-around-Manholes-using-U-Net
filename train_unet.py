import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from tqdm import tqdm

DATASET_ROOT = '/path/to/dataset'  # Update this path to your dataset root
CHECKPOINT_DIR = '/path/to/checkpoints'  # Update this path to your checkpoints directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class CrackDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        img_dir = os.path.join(root, split, 'images')
        mask_dir = os.path.join(root, split, 'masks')
        
        self.images = sorted([os.path.join(img_dir, f) 
                             for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.masks = sorted([os.path.join(mask_dir, f) 
                            for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)


def get_transform(split='train'):
    """
    Augmentations are applied through roboflow platform. So, only basic transforms here.
    """
    if split == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.CLAHE(clip_limit=3.0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class FocalDiceLoss(nn.Module):
    """
    Focal Loss - for class imbalance
    Dice Loss - Segmentation quality improvement
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        # Focal Loss
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pred_sigmoid = torch.sigmoid(pred)
        
        # pt: probability of the true class
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        
        # Focal weight: high weight on hard examples
        focal_weight = (
            (self.alpha * target + (1 - self.alpha) * (1 - target)) * 
            ((1 - pt) ** self.gamma)
        )
        focal_loss = (focal_weight * bce).mean()
        
        # Dice Loss
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        # Combined
        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        
        return total_loss


def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    
    # IoU
    intersection = (pred_binary * target).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    
    # Dice
    dice = (2 * intersection + 1e-5) / (pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-5)
    
    return iou.mean().item(), dice.mean().item()


def train_model(num_epochs=100, batch_size=8, lr=1e-4, device='cuda'):
    print("U-Net Training - Manhole Crack Segmentation...")

    log_path = os.path.join(CHECKPOINT_DIR, 'training_log.txt')
    log_file = open(log_path, 'w')
    
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print("U-Net Training - Manhole Crack Segmentation")
    log_print("="*80)
    
    # Dataset
    train_dataset = CrackDataset(DATASET_ROOT, 'train', get_transform('train'))
    val_dataset = CrackDataset(DATASET_ROOT, 'val', get_transform('val'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    log_print(f"Train: {len(train_dataset)} images")
    log_print(f"Val: {len(val_dataset)} images")
    log_print(f"Batch size: {batch_size}")
    log_print(f"Learning rate: {lr}")
    
    # Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    
    print(f"Model: U-Net with ResNet34 encoder")
    
    # Loss & Optimizer
    criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=7, factor=0.5
    )
    
    best_iou = 0.0
    patience_counter = 0
    max_patience = 20
    
    log_print("\n" + "="*80)
    log_print("Starting training...")
    log_print("="*80)
    
    for epoch in range(num_epochs):
        log_print(f"\nEpoch {epoch+1}/{num_epochs}")
        log_print("-" * 80)
        
        # Train
        model.train()
        train_loss = 0
        train_iou = 0
        
        for images, masks in tqdm(train_loader, desc='Training'):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                pred = torch.sigmoid(outputs)
                iou, _ = calculate_metrics(pred, masks)
                train_iou += iou
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                pred = torch.sigmoid(outputs)
                iou, dice = calculate_metrics(pred, masks)
                val_iou += iou
                val_dice += dice
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Print metrics
        log_print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        log_print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            log_print(f"Saved best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            log_print(f"No improvement ({patience_counter}/{max_patience})")
        
        # Early stopping
        if patience_counter >= max_patience:
            log_print(f"\nEarly stopping at epoch {epoch+1}")  
            break
        
        # Learning rate scheduling
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        log_print(f"Learning rate: {current_lr:.2e}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    log_print("\n" + "="*80)
    log_print("Training Complete!")
    log_print("="*80)
    log_print(f"Best Validation IoU: {best_iou:.4f}")
    log_print(f"Model saved to: {CHECKPOINT_DIR}/best_model.pth")
    
    log_file.close()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_model(
        num_epochs=100,
        batch_size=8,
        lr=5e-5,
        device=device
    )

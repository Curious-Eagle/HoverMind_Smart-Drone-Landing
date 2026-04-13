import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm import tqdm
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import segmentation_models_pytorch as smp
    from torchsummary import summary
except ImportError:
    logger.error("Required packages not installed. Please install segmentation-models-pytorch and torchsummary")
    raise

class Config:
    """Configuration class for the drone landing segmentation project"""
    def __init__(self):
        self.n_classes = 23
        self.batch_size = 4  # Increased batch size for better GPU utilization
        self.max_lr = 1e-3
        self.epochs = 20  # Increased epochs
        self.weight_decay = 1e-4
        self.early_stopping_patience = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_size = (704, 1056)
        self.test_size = 0.15
        self.val_size = 0.15
        self.random_state = 42

class ImprovedDroneDataset(Dataset):
    """Enhanced dataset class with better error handling and caching"""
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, cache_data=False):
        self.img_path = Path(img_path)
        self.mask_path = Path(mask_path)
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.cache_data = cache_data
        self.cache = {}
        
        # Validate paths
        if not self.img_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {self.img_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask path does not exist: {self.mask_path}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
            
        try:
            img_file = self.img_path / f"{self.X[idx]}.jpg"
            mask_file = self.mask_path / f"{self.X[idx]}.png"
            
            if not img_file.exists():
                raise FileNotFoundError(f"Image file not found: {img_file}")
            if not mask_file.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_file}")
            
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if self.transform is not None:
                aug = self.transform(image=img, mask=mask)
                img = Image.fromarray(aug['image'])
                mask = aug['mask']
            else:
                img = Image.fromarray(img)
            
            # Transform differently for image and mask
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
            img = t(img)
            mask = torch.from_numpy(mask).long()
            
            result = (img, mask)
            
            if self.cache_data:
                self.cache[idx] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error loading data at index {idx}: {str(e)}")
            raise

class ModelTrainer:
    """Enhanced training class with better monitoring and early stopping"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': [],
            'train_acc': [], 'val_acc': [], 'lrs': []
        }
        
    def pixel_accuracy(self, output, mask):
        with torch.no_grad():
            output = torch.argmax(F.softmax(output, dim=1), dim=1)
            correct = torch.eq(output, mask).int()
            accuracy = float(correct.sum()) / float(correct.numel())
        return accuracy

    def compute_miou(self, pred_mask, mask, smooth=1e-10):
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.contiguous().view(-1)
            mask = mask.contiguous().view(-1)

            iou_per_class = []
            for cls in range(self.config.n_classes):
                true_class = pred_mask == cls
                true_label = mask == cls
                
                if true_label.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    union = torch.logical_or(true_class, true_label).sum().float().item()
                    iou = (intersect + smooth) / (union + smooth)
                    iou_per_class.append(iou)
            
            return np.nanmean(iou_per_class)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_epoch(self, model, train_loader, criterion, optimizer, scheduler):
        model.train()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        
        for i, (image, mask) in enumerate(tqdm(train_loader, desc="Training")):
            image = image.to(self.device)
            mask = mask.to(self.device)
            
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, mask)
            
            # Calculate metrics
            iou_score += self.compute_miou(output, mask)
            accuracy += self.pixel_accuracy(output, mask)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            self.history['lrs'].append(self.get_lr(optimizer))
            
        return running_loss / len(train_loader), iou_score / len(train_loader), accuracy / len(train_loader)

    def validate_epoch(self, model, val_loader, criterion):
        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0
        
        with torch.no_grad():
            for image, mask in tqdm(val_loader, desc="Validation"):
                image = image.to(self.device)
                mask = mask.to(self.device)
                
                output = model(image)
                val_iou_score += self.compute_miou(output, mask)
                test_accuracy += self.pixel_accuracy(output, mask)
                loss = criterion(output, mask)
                test_loss += loss.item()
                
        return test_loss / len(val_loader), val_iou_score / len(val_loader), test_accuracy / len(val_loader)

    def fit(self, model, train_loader, val_loader, criterion, optimizer, scheduler, save_path="best_model.pt"):
        torch.cuda.empty_cache()
        min_loss = np.inf
        best_miou = 0
        patience_counter = 0
        
        model.to(self.device)
        fit_time = time.time()
        
        for epoch in range(self.config.epochs):
            since = time.time()
            
            # Training
            train_loss, train_iou, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, scheduler)
            
            # Validation
            val_loss, val_iou, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_miou'].append(train_iou)
            self.history['val_miou'].append(val_iou)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model based on IoU
            if val_iou > best_miou:
                best_miou = val_iou
                logger.info(f'Best IoU improved: {best_miou:.3f}. Saving model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_iou': val_iou,
                    'val_loss': val_loss,
                }, save_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            epoch_time = (time.time() - since) / 60
            logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                       f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, "
                       f"Train IoU: {train_iou:.3f}, Val IoU: {val_iou:.3f}, "
                       f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, "
                       f"Time: {epoch_time:.2f}m")
        
        total_time = (time.time() - fit_time) / 60
        logger.info(f'Total training time: {total_time:.2f} minutes')
        return self.history

class Visualizer:
    """Enhanced visualization class with better plots"""
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['val_loss'], label='Validation', marker='o', linewidth=2)
        axes[0, 0].plot(history['train_loss'], label='Training', marker='o', linewidth=2)
        axes[0, 0].set_title('Loss per Epoch', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU plot
        axes[0, 1].plot(history['train_miou'], label='Training mIoU', marker='*', linewidth=2)
        axes[0, 1].plot(history['val_miou'], label='Validation mIoU', marker='*', linewidth=2)
        axes[0, 1].set_title('IoU Score per Epoch', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Mean IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1, 0].plot(history['train_acc'], label='Training Accuracy', marker='*', linewidth=2)
        axes[1, 0].plot(history['val_acc'], label='Validation Accuracy', marker='*', linewidth=2)
        axes[1, 0].set_title('Accuracy per Epoch', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 1].plot(history['lrs'], linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def visualize_predictions(model, test_set, indices, config, save_path=None):
        model.eval()
        fig, axes = plt.subplots(len(indices), 3, figsize=(20, 6 * len(indices)))
        
        if len(indices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            image, mask = test_set[idx]
            
            # Predict
            t = T.Compose([T.ToTensor(), T.Normalize(config.mean, config.std)])
            image_tensor = t(image)
            image_tensor = image_tensor.unsqueeze(0).to(config.device)
            
            with torch.no_grad():
                output = model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).cpu().squeeze(0)
            
            # Display
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Original Image {idx}', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask, cmap='tab20')
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='tab20')
            axes[i, 2].set_title('Prediction', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_data_loaders(image_path, mask_path, config):
    """Create optimized data loaders with enhanced augmentations"""
    
    # Create dataframe
    names = []
    for file_path in Path(image_path).glob("*.jpg"):
        names.append(file_path.stem)
    
    df = pd.DataFrame({'id': names}, index=np.arange(len(names)))
    logger.info(f'Total images found: {len(df)}')
    
    # Split dataset
    X_trainval, X_test = train_test_split(
        df['id'].values, 
        test_size=config.test_size, 
        random_state=config.random_state
    )
    X_train, X_val = train_test_split(
        X_trainval, 
        test_size=config.val_size, 
        random_state=config.random_state
    )
    
    logger.info(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    
    # Enhanced transformations
    train_transforms = A.Compose([
        A.Resize(*config.image_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    ])
    
    val_transforms = A.Compose([
        A.Resize(*config.image_size, interpolation=cv2.INTER_NEAREST),
    ])
    
    test_transforms = A.Compose([
        A.Resize(*config.image_size, interpolation=cv2.INTER_NEAREST),
    ])
    
    # Create datasets
    train_set = ImprovedDroneDataset(
        image_path, mask_path, X_train, config.mean, config.std, train_transforms
    )
    val_set = ImprovedDroneDataset(
        image_path, mask_path, X_val, config.mean, config.std, val_transforms
    )
    test_set = ImprovedDroneDataset(
        image_path, mask_path, X_test, config.mean, config.std, test_transforms
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader, test_set

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    config = Config()
    logger.info(f"Using device: {config.device}")
    
    # Data paths (update these to your actual paths)
    IMAGE_PATH = "data/images/"
    MASK_PATH = "data/masks/"
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader, test_set = create_data_loaders(
            IMAGE_PATH, MASK_PATH, config
        )
        
        # Initialize model with improved architecture
        model = smp.Unet(
            encoder_name='efficientnet-b3',  # Better encoder
            encoder_weights='imagenet',
            classes=config.n_classes,
            activation=None,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16]
        )
        
        # Initialize training components
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.max_lr, 
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.max_lr, 
            epochs=config.epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Train model
        trainer = ModelTrainer(config)
        history = trainer.fit(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            save_path="best_drone_segmentation_model.pt"
        )
        
        # Visualize training results
        visualizer = Visualizer()
        visualizer.plot_training_history(history, save_path="training_history.png")
        
        # Load best model for testing
        checkpoint = torch.load("best_drone_segmentation_model.pt", map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Visualize test predictions
        test_indices = [0, 5, 10, 15, 20]  # Sample test indices
        visualizer.visualize_predictions(
            model, test_set, test_indices, config, 
            save_path="test_predictions.png"
        )
        
        # Save training history
        with open("training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
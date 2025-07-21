import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import librosa
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import logging
import json
from datetime import datetime
import time
import boto3
import argparse
from sklearn.model_selection import train_test_split

# -------------------------------
# ENHANCED LOGGING SETUP
# -------------------------------
def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__), log_file

# -------------------------------
# PROGRESS TRACKING
# -------------------------------
class TrainingTracker:
    def __init__(self, log_file):
        self.log_file = log_file
        self.training_stats = {
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'best_loss': float('inf'),
            'total_iterations': 0,
            'total_time': 0
        }
        self.epoch_start_time = None
    
    def start_epoch(self, epoch):
        self.epoch_start_time = time.time()
        logging.info(f"Starting Epoch {epoch}")
    
    def log_iteration(self, epoch, iteration, loss, learning_rate):
        if iteration % 10 == 0:  # Log every 10 iterations
            elapsed = time.time() - self.epoch_start_time
            logging.info(f"Epoch {epoch}, Iteration {iteration}, Loss: {loss:.4f}, LR: {learning_rate:.6f}, Time: {elapsed:.2f}s")
    
    def end_epoch(self, epoch, avg_loss, learning_rate):
        epoch_time = time.time() - self.epoch_start_time
        self.training_stats['epochs'].append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'learning_rate': learning_rate,
            'time': epoch_time
        })
        
        logging.info(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Save progress to JSON file
        self.save_progress()
    
    def save_progress(self):
        progress_file = self.log_file.replace('.log', '_progress.json')
        with open(progress_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def log_model_save(self, filename):
        logging.info(f"Model saved: {filename}")

# -------------------------------
# COST-OPTIMIZED CONFIGURATION
# -------------------------------
VIDEO_DIRS = {'ads': 'data/ads', 'games': 'data/games'}
VAL_VIDEO_DIRS = {'ads': 'data/validation/ads', 'games': 'data/validation/games'}  # Separate validation data
CLIP_DURATION = 10  # seconds
FPS = 4  # Reduced from 8 to save memory and speed
NUM_FRAMES = CLIP_DURATION * FPS  # 40 frames instead of 80
FRAME_SIZE = (96, 96)  # Reduced from 112x112 to save memory
MFCC_N_MELS = 32  # Reduced from 40 to save memory
BATCH_SIZE = 8  # Increased batch size for efficiency
EPOCHS = 8  # Reduced epochs, use early stopping
LEARNING_RATE = 2e-4  # Slightly higher for faster convergence
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable mixed precision for 2x speedup and memory savings
USE_AMP = True

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'your-ad-game-data-bucket')
S3_RESULTS_PREFIX = 'results'

# -------------------------------
# MEMORY-EFFICIENT DATASET
# -------------------------------
class CostOptimizedDataset(Dataset):
    def __init__(self, video_dirs, transform=None):
        self.samples = []
        self.transform = transform
        for label, path in enumerate(video_dirs.values()):
            video_path = path+"/video"
            audio_path = path+"/audio"
            video_files = sorted(glob(os.path.join(video_path, "*.mp4")))
            for vf in video_files:
                base = os.path.splitext(os.path.basename(vf))[0]
                wav_path = os.path.join(audio_path, base + ".wav")
                if os.path.exists(wav_path):
                    self.samples.append((vf, wav_path, label))
        
        logging.info(f"Dataset loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, audio_path, label = self.samples[idx]

        # Memory-efficient frame loading
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Simple uniform sampling
        step = max(total_frames // NUM_FRAMES, 1)
        for i in range(NUM_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, FRAME_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Use zero frame if reading fails
                frames.append(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
        
        cap.release()

        # Convert to tensor efficiently
        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = torch.tensor(frames).permute(0, 3, 1, 2)

        # Simplified audio processing
        y, sr = librosa.load(audio_path, sr=16000)  # Fixed sample rate
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_MELS, hop_length=1024)
        mfcc = np.mean(mfcc, axis=1)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        return frames, mfcc, torch.tensor(label, dtype=torch.float32)

# -------------------------------
# VALIDATION FUNCTIONS
# -------------------------------
def create_validation_dataset():
    """Create validation dataset from separate validation data directory"""
    try:
        val_dataset = CostOptimizedDataset(VAL_VIDEO_DIRS)
        logging.info(f"Validation dataset loaded: {len(val_dataset)} samples")
        return val_dataset
    except Exception as e:
        logging.warning(f"Validation dataset not found: {e}")
        logging.info("Falling back to using 20% of training data for validation")
        return None

def create_train_val_split_fallback(dataset, val_split=0.2, random_state=42):
    """Fallback: Split dataset into train and validation sets if no separate validation data"""
    total_samples = len(dataset)
    indices = list(range(total_samples))
    
    # Get labels for stratification
    labels = [dataset.samples[i][2] for i in indices]
    
    # Stratified split to maintain class balance
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=random_state,
        stratify=labels
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    logging.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset

def validate_epoch(model, val_dataloader, criterion, device):
    """Run validation for one epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, audios, labels in tqdm(val_dataloader, desc="Validation"):
            videos, audios, labels = videos.to(device), audios.to(device), labels.to(device)
            
            outputs = model(videos, audios)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_val_loss, accuracy

# -------------------------------
# S3 UPLOAD FUNCTIONS
# -------------------------------
def upload_to_s3(local_file, s3_key, bucket_name=None):
    """Upload file to S3"""
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_file, bucket_name, s3_key)
        logging.info(f"Uploaded {local_file} to s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        logging.error(f"Failed to upload {local_file} to S3: {e}")
        return False

def upload_training_results(model_files, log_files, bucket_name=None):
    """Upload training results to S3"""
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_prefix = f"{S3_RESULTS_PREFIX}/{timestamp}"
    
    uploaded_files = []
    
    # Upload model files
    for model_file in model_files:
        if os.path.exists(model_file):
            s3_key = f"{results_prefix}/models/{os.path.basename(model_file)}"
            if upload_to_s3(model_file, s3_key, bucket_name):
                uploaded_files.append(s3_key)
    
    # Upload log files
    for log_file in log_files:
        if os.path.exists(log_file):
            s3_key = f"{results_prefix}/logs/{os.path.basename(log_file)}"
            if upload_to_s3(log_file, s3_key, bucket_name):
                uploaded_files.append(s3_key)
    
    return uploaded_files

# -------------------------------
# CHECKPOINT FUNCTIONS
# -------------------------------
def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load training checkpoint or best model"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        # Check if this is a full checkpoint or just a model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with optimizer state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            logging.info(f"Full checkpoint loaded: {filename} (Epoch {epoch}, Loss {loss:.4f})")
            return epoch, loss
        else:
            # Best model file (model state dict only)
            model.load_state_dict(checkpoint)
            logging.info(f"Best model loaded: {filename} (model only, starting from epoch 1)")
            return 0, float('inf')  # Start from epoch 1
    else:
        logging.warning(f"Checkpoint not found: {filename}")
        return 0, float('inf')

# -------------------------------
# LIGHTWEIGHT MODEL (Updated)
# -------------------------------
class CostOptimizedClassifier(nn.Module):
    def __init__(self):
        super(CostOptimizedClassifier, self).__init__()
        # Smaller video backbone
        base_model = models.mobilenet_v2(pretrained=True)
        self.video_cnn = nn.Sequential(
            *list(base_model.features),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.video_fc = nn.Linear(1280, 128)

        # Smaller audio branch
        self.audio_fc = nn.Sequential(
            nn.Linear(MFCC_N_MELS, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Simple fusion - remove sigmoid since we're using BCEWithLogitsLoss
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # Removed nn.Sigmoid() since BCEWithLogitsLoss handles it
        )

    def forward(self, video, audio):
        B, T, C, H, W = video.shape
        video = video.view(B * T, C, H, W)
        v_feat = self.video_cnn(video).view(B, T, -1).mean(dim=1)
        v_feat = self.video_fc(v_feat)
        a_feat = self.audio_fc(audio)
        x = torch.cat([v_feat, a_feat], dim=1)
        return self.classifier(x).squeeze()

# -------------------------------
# COST-OPTIMIZED TRAINING (Enhanced)
# -------------------------------
def train_cost_optimized(resume_from=None, upload_to_s3_flag=False):
    # Setup logging
    logger, log_file = setup_logging()
    tracker = TrainingTracker(log_file)
    
    logger.info("Starting cost-optimized training with validation...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Resume from: {resume_from}")
    logger.info(f"Upload to S3: {upload_to_s3_flag}")
    logger.info(f"Training data: {VIDEO_DIRS}")
    logger.info(f"Validation data: {VAL_VIDEO_DIRS}")
    
    # Load training dataset (all data since model has already seen it)
    logger.info("Loading training dataset...")
    train_dataset = CostOptimizedDataset(VIDEO_DIRS)
    
    # Try to load separate validation dataset
    logger.info("Loading validation dataset...")
    val_dataset = create_validation_dataset()
    
    if val_dataset is None:
        # Fallback: split training data if no separate validation data
        logger.info("No separate validation data found. Using 20% of training data for validation.")
        train_dataset, val_dataset = create_train_val_split_fallback(train_dataset, val_split=0.2, random_state=42)
    else:
        # Use all training data since model has already seen it
        logger.info("Using separate validation dataset. Training on all available data.")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    # Initialize model and training components
    logger.info("Initializing model...")
    model = CostOptimizedClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_epoch, best_train_loss = load_checkpoint(model, optimizer, resume_from)
        logger.info(f"Resumed from epoch {start_epoch} with loss {best_train_loss:.4f}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        epochs=EPOCHS, 
        steps_per_epoch=len(train_dataloader)
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler() if USE_AMP and torch.cuda.is_available() else None

    logger.info("Starting training loop...")
    
    for epoch in range(start_epoch, EPOCHS):
        tracker.start_epoch(epoch + 1)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for iteration, (videos, audios, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            if USE_AMP and scaler is not None:
                with torch.amp.autocast():
                    outputs = model(videos, audios)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(videos, audios)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()
            
            # Track progress
            current_lr = scheduler.get_last_lr()[0]
            tracker.log_iteration(epoch + 1, iteration + 1, loss.item(), current_lr)

        avg_train_loss = running_loss / len(train_dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation phase
        logger.info(f"Running validation for epoch {epoch+1}...")
        avg_val_loss, val_accuracy = validate_epoch(model, val_dataloader, criterion, DEVICE)
        
        logger.info(f"Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        tracker.end_epoch(epoch + 1, avg_train_loss, current_lr)
        
        # Save checkpoint
        checkpoint_filename = f"cost_optimized_checkpoint_epoch_{epoch+1}.pth"
        save_checkpoint(model, optimizer, epoch + 1, avg_val_loss, checkpoint_filename)
        
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model_filename = f"cost_optimized_best_epoch_{epoch+1}_val.pth"
            torch.save(model.state_dict(), model_filename)
            tracker.log_model_save(model_filename)
            logger.info(f"New best model saved: {model_filename} (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save final model
    final_model_filename = "cost_optimized_final_with_validation.pth"
    torch.save(model.state_dict(), final_model_filename)
    tracker.log_model_save(final_model_filename)
    
    # Upload results to S3 if requested
    if upload_to_s3_flag:
        logger.info("Uploading training results to S3...")
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        log_files = [log_file, log_file.replace('.log', '_progress.json')]
        uploaded_files = upload_training_results(model_files, log_files)
        logger.info(f"Uploaded {len(uploaded_files)} files to S3")
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Progress file: {log_file.replace('.log', '_progress.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ad/game classifier with validation')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--upload-s3', action='store_true', help='Upload results to S3')
    parser.add_argument('--s3-bucket', type=str, help='S3 bucket name for uploads')
    
    args = parser.parse_args()
    
    # Set S3 bucket if provided
    if args.s3_bucket:
        S3_BUCKET_NAME = args.s3_bucket
    
    # Check for existing checkpoint
    if not args.resume:
        # Look for the most recent checkpoint (prefer full checkpoints over best models)
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('cost_optimized_checkpoint_') and f.endswith('.pth')]
        best_model_files = [f for f in os.listdir('.') if f.startswith('cost_optimized_best_epoch_') and f.endswith('.pth')]
        
        if checkpoint_files:
            # Prefer full checkpoints for resuming
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            args.resume = checkpoint_files[-1]
            print(f"Found checkpoint: {args.resume}")
        elif best_model_files:
            # Fallback to best model files
            best_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            args.resume = best_model_files[-1]
            print(f"Found best model: {args.resume}")
            print("Note: This is a best model file (model only). Training will start from epoch 1.")
    
    train_cost_optimized(resume_from=args.resume, upload_to_s3_flag=args.upload_s3)
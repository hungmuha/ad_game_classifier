import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import librosa
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import logging
import json
from datetime import datetime
import time

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
def train_cost_optimized():
    # Setup logging
    logger, log_file = setup_logging()
    tracker = TrainingTracker(log_file)
    
    logger.info("Starting cost-optimized training...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = CostOptimizedDataset(VIDEO_DIRS)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Steps per epoch: {len(dataloader)}")

    # Initialize model and training components
    logger.info("Initializing model...")
    model = CostOptimizedClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        epochs=EPOCHS, 
        steps_per_epoch=len(dataloader)
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    logger.info("Starting training loop...")
    
    for epoch in range(EPOCHS):
        tracker.start_epoch(epoch + 1)
        running_loss = 0.0
        
        for iteration, (videos, audios, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            if USE_AMP:
                with torch.amp.autocast('cuda'):
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

        avg_loss = running_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        tracker.end_epoch(epoch + 1, avg_loss, current_lr)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model_filename = f"cost_optimized_best_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_filename)
            tracker.log_model_save(model_filename)
            logger.info(f"New best model saved: {model_filename}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save final model
    final_model_filename = "cost_optimized_final.pth"
    torch.save(model.state_dict(), final_model_filename)
    tracker.log_model_save(final_model_filename)
    
    logger.info("Training completed successfully!")
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Progress file: {log_file.replace('.log', '_progress.json')}")

if __name__ == "__main__":
    train_cost_optimized()
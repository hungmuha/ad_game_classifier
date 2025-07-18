import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import librosa
import cv2
import numpy as np
import boto3
import tempfile
import io
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# S3 STREAMING CONFIGURATION
# -------------------------------
S3_BUCKET = "nfl-ads-training"  # Replace with your bucket
S3_DATA_PREFIX = "data"
CLIP_DURATION = 10  # seconds
FPS = 4  # Reduced for efficiency
NUM_FRAMES = CLIP_DURATION * FPS  # 40 frames
FRAME_SIZE = (96, 96)  # Reduced size
MFCC_N_MELS = 32  # Reduced features
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True

# S3 client setup
s3_client = boto3.client('s3')

# -------------------------------
# S3 STREAMING DATASET
# -------------------------------
class S3StreamingDataset(IterableDataset):
    def __init__(self, s3_bucket, s3_prefix, transform=None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        self.samples = self._list_s3_files()
        logger.info(f"Found {len(self.samples)} samples in S3")
        
    def _list_s3_files(self):
        """List all video files in S3 and create sample pairs"""
        samples = []
        
        # List ads videos
        ads_videos = self._list_s3_objects(f"{self.s3_prefix}/ads/video/", ".mp4")
        for video_key in ads_videos:
            audio_key = video_key.replace("/video/", "/audio/").replace(".mp4", ".wav")
            if self._s3_object_exists(audio_key):
                samples.append((video_key, audio_key, 0))  # 0 = ads
        
        # List games videos
        games_videos = self._list_s3_objects(f"{self.s3_prefix}/games/video/", ".mp4")
        for video_key in games_videos:
            audio_key = video_key.replace("/video/", "/audio/").replace(".mp4", ".wav")
            if self._s3_object_exists(audio_key):
                samples.append((video_key, audio_key, 1))  # 1 = games
        
        return samples
    
    def _list_s3_objects(self, prefix, extension):
        """List S3 objects with specific prefix and extension"""
        objects = []
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith(extension):
                        objects.append(obj['Key'])
        
        return objects
    
    def _s3_object_exists(self, key):
        """Check if S3 object exists"""
        try:
            s3_client.head_object(Bucket=self.s3_bucket, Key=key)
            return True
        except:
            return False
    
    def _download_s3_object(self, key):
        """Download S3 object to temporary file"""
        try:
            response = s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return None
    
    def _process_video_from_s3(self, video_key):
        """Process video directly from S3 without downloading to disk"""
        video_data = self._download_s3_object(video_key)
        if video_data is None:
            return None
        
        # Create temporary file for OpenCV
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_data)
            temp_file.flush()
            
            # Process video
            cap = cv2.VideoCapture(temp_file.name)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Uniform sampling
            step = max(total_frames // NUM_FRAMES, 1)
            for i in range(NUM_FRAMES):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))
            
            cap.release()
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return frames
    
    def _process_audio_from_s3(self, audio_key):
        """Process audio directly from S3"""
        audio_data = self._download_s3_object(audio_key)
        if audio_data is None:
            return None
        
        # Process audio from memory
        audio_bytes = io.BytesIO(audio_data)
        y, sr = librosa.load(audio_bytes, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_MELS, hop_length=1024)
        mfcc = np.mean(mfcc, axis=1)
        return mfcc
    
    def __iter__(self):
        """Stream samples from S3"""
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single worker
            samples = self.samples
        else:
            # Multiple workers - split samples
            per_worker = len(self.samples) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.samples)
            samples = self.samples[start:end]
        
        for video_key, audio_key, label in samples:
            # Process video from S3
            frames = self._process_video_from_s3(video_key)
            if frames is None:
                continue
            
            # Process audio from S3
            mfcc = self._process_audio_from_s3(audio_key)
            if mfcc is None:
                continue
            
            # Convert to tensors
            frames = np.stack(frames).astype(np.float32) / 255.0
            frames = torch.tensor(frames).permute(0, 3, 1, 2)
            mfcc = torch.tensor(mfcc, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            yield frames, mfcc, label_tensor

# -------------------------------
# LIGHTWEIGHT MODEL (Same as before)
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

        # Simple fusion
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
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
# S3 STREAMING TRAINING
# -------------------------------
def train_s3_streaming():
    logger.info("Initializing S3 streaming dataset...")
    dataset = S3StreamingDataset(S3_BUCKET, S3_DATA_PREFIX)
    
    # Use IterableDataset with DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=2,  # Reduced for S3 streaming
        pin_memory=True
    )

    model = CostOptimizedClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        epochs=EPOCHS, 
        steps_per_epoch=len(dataloader)
    )

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    logger.info("Starting S3 streaming training...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        batch_count = 0
        
        for videos, audios, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
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
            batch_count += 1

        avg_loss = running_loss / batch_count
        logger.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "s3_streaming_best.pth")
            logger.info("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), "s3_streaming_final.pth")
    logger.info("S3 streaming training complete. Models saved.")

if __name__ == "__main__":
    train_s3_streaming() 
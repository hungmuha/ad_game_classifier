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
# LIGHTWEIGHT MODEL
# -------------------------------
class CostOptimizedClassifier(nn.Module):
    def __init__(self):
        super(CostOptimizedClassifier, self).__init__()
        # Smaller video backbone
        base_model = models.mobilenet_v2(pretrained=True)
        # Remove some layers to make it lighter
        self.video_cnn = nn.Sequential(
            *list(base_model.features)[:-4],  # Remove last 4 layers
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.video_fc = nn.Linear(320, 128)  # Reduced from 1280->256 to 320->128

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
        v_feat = self.video_cnn(video).view(B, T, -1).mean(dim=1)  # Simple temporal pooling
        v_feat = self.video_fc(v_feat)
        a_feat = self.audio_fc(audio)
        x = torch.cat([v_feat, a_feat], dim=1)
        return self.classifier(x).squeeze()

# -------------------------------
# COST-OPTIMIZED TRAINING
# -------------------------------
def train_cost_optimized():
    dataset = CostOptimizedDataset(VIDEO_DIRS)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,  # More workers for faster data loading
        pin_memory=True  # Faster GPU transfer
    )

    model = CostOptimizedClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler for faster convergence
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
    patience = 3  # Early stopping
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
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

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "cost_optimized_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    torch.save(model.state_dict(), "cost_optimized_final.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_cost_optimized() 

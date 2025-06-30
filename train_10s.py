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
# CONFIGURATION FOR 10-SECOND CLIPS
# -------------------------------
VIDEO_DIRS = {'ads': 'data/ads', 'games': 'data/games'}
CLIP_DURATION = 10  # seconds
FPS = 8  # 8 FPS for better temporal coverage
NUM_FRAMES = CLIP_DURATION * FPS  # 80 frames
FRAME_SIZE = (112, 112)
MFCC_N_MELS = 40
BATCH_SIZE = 4  # Can be higher with optimized memory usage
EPOCHS = 15  # More epochs since fewer samples
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# OPTIMIZED DATASET FOR 10-SECOND CLIPS
# -------------------------------
class VideoAudioDataset10s(Dataset):
    def __init__(self, video_dirs, transform=None):
        self.samples = []
        self.transform = transform
        for label, path in enumerate(video_dirs.values()):
            video_files = sorted(glob(os.path.join(path, "*.mp4")))
            for vf in video_files:
                base = os.path.splitext(os.path.basename(vf))[0]
                wav_path = os.path.join(path, base + ".wav")
                if os.path.exists(wav_path):
                    self.samples.append((vf, wav_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, audio_path, label = self.samples[idx]

        # Optimized frame sampling for 10-second clips
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Better sampling strategy: sample evenly across the 10 seconds
        if total_frames >= NUM_FRAMES:
            # If we have enough frames, sample evenly
            step = total_frames // NUM_FRAMES
            for i in range(NUM_FRAMES):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        else:
            # If not enough frames, repeat the last frame
            for i in range(NUM_FRAMES):
                frame_idx = min(i, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        
        cap.release()

        # Pad if not enough frames
        while len(frames) < NUM_FRAMES:
            frames.append(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8))

        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Audio processing optimized for 10-second clips
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCC with better temporal resolution
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_MELS, hop_length=512)
        
        # Average over time to get a single feature vector per clip
        mfcc = np.mean(mfcc, axis=1)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        return frames, mfcc, torch.tensor(label, dtype=torch.float32)

# -------------------------------
# ENHANCED MODEL FOR 10-SECOND CLIPS
# -------------------------------
class MultimodalClassifier10s(nn.Module):
    def __init__(self):
        super(MultimodalClassifier10s, self).__init__()
        # Video branch with temporal attention
        base_model = models.mobilenet_v2(pretrained=True)
        self.video_cnn = nn.Sequential(*list(base_model.features), nn.AdaptiveAvgPool2d((1, 1)))
        self.video_fc = nn.Linear(1280, 256)
        
        # Temporal attention for better 10-second modeling
        self.temporal_attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        # Audio branch
        self.audio_fc = nn.Sequential(
            nn.Linear(MFCC_N_MELS, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # Fusion with better architecture
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        B, T, C, H, W = video.shape
        video = video.view(B * T, C, H, W)
        v_feat = self.video_cnn(video).view(B, T, -1)  # (B, T, 1280)
        v_feat = self.video_fc(v_feat)  # (B, T, 256)
        
        # Apply temporal attention
        v_feat_attended, _ = self.temporal_attention(v_feat, v_feat, v_feat)
        v_feat = v_feat_attended.mean(dim=1)  # (B, 256)
        
        a_feat = self.audio_fc(audio)
        x = torch.cat([v_feat, a_feat], dim=1)
        return self.classifier(x).squeeze()

# -------------------------------
# TRAINING LOOP
# -------------------------------
def train_10s():
    dataset = VideoAudioDataset10s(VIDEO_DIRS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = MultimodalClassifier10s().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    model.train()
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for videos, audios, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(videos, audios)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "ad_vs_game_classifier_10s_best.pth")

    torch.save(model.state_dict(), "ad_vs_game_classifier_10s_final.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_10s() 

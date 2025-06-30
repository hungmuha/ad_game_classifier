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
# CONFIGURATION
# -------------------------------
# Define configuration parameters
VIDEO_DIRS = {'ads': 'data/ads', 'games': 'data/games'}
CLIP_DURATION = 10  # seconds - updated to match your data
FPS = 6  # increased from 3 to get better temporal coverage for 10-second clips
NUM_FRAMES = CLIP_DURATION * FPS  # 60 frames instead of 15
FRAME_SIZE = (112, 112)
MFCC_N_MELS = 40
BATCH_SIZE = 2  # reduced due to increased memory usage
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# DATASET DEFINITION
# -------------------------------
class VideoAudioDataset(Dataset):
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

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // NUM_FRAMES, 1)
        for i in range(NUM_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
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

        # Load audio and extract MFCC
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_MELS)
        mfcc = np.mean(mfcc, axis=1)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)

        return frames, mfcc, torch.tensor(label, dtype=torch.float32)

# -------------------------------
# MODEL DEFINITION
# -------------------------------
class MultimodalClassifier(nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        # Video branch
        base_model = models.mobilenet_v2(pretrained=True)
        self.video_cnn = nn.Sequential(*list(base_model.features), nn.AdaptiveAvgPool2d((1, 1)))
        self.video_fc = nn.Linear(1280, 256)

        # Audio branch
        self.audio_fc = nn.Sequential(
            nn.Linear(MFCC_N_MELS, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 64),
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
# TRAINING LOOP
# -------------------------------
def train():
    dataset = VideoAudioDataset(VIDEO_DIRS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultimodalClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
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

        print(f"Epoch {epoch+1} Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "ad_vs_game_classifier.pth")
    print("Training complete. Model saved to ad_vs_game_classifier.pth")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    train()


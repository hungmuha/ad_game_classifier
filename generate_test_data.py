import os
import numpy as np
import cv2
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile

# -------------------------------
# GENERATE TEST DATA
# -------------------------------
def create_test_video(filename, duration=10, fps=29, width=640, height=480):
    """Create a synthetic test video."""
    # Create frames
    frames = []
    for i in range(duration * fps):
        # Create a frame with some movement
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some moving elements
        x = int((i / (duration * fps)) * width)
        y = height // 2
        
        # Draw a moving rectangle
        cv2.rectangle(frame, (x-50, y-50), (x+50, y+50), (255, 0, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    # Save as video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    return filename

def create_test_audio(filename, duration=10, sample_rate=16000):
    """Create a synthetic test audio."""
    # Generate some audio (sine wave with noise)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple tone
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise
    noise = 0.1 * np.random.randn(len(audio))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save as WAV
    librosa.output.write_wav(filename, audio, sample_rate)
    
    return filename

def generate_test_dataset():
    """Generate a complete test dataset."""
    print("🎬 Generating test dataset...")
    
    # Create directories
    os.makedirs('data/ads', exist_ok=True)
    os.makedirs('data/games', exist_ok=True)
    
    # Generate test videos and audio
    for i in range(5):  # 5 samples per class
        # Generate ad samples
        ad_video = f'data/ads/ad_test_{i}.mp4'
        ad_audio = f'data/ads/ad_test_{i}.wav'
        
        create_test_video(ad_video, duration=10)
        create_test_audio(ad_audio, duration=10)
        
        print(f"Created ad sample {i+1}/5")
        
        # Generate game samples
        game_video = f'data/games/game_test_{i}.mp4'
        game_audio = f'data/games/game_test_{i}.wav'
        
        create_test_video(game_video, duration=10)
        create_test_audio(game_audio, duration=10)
        
        print(f"Created game sample {i+1}/5")
    
    print("✅ Test dataset generated successfully!")
    print("📁 Files created:")
    print("  - data/ads/: 5 MP4 + 5 WAV files")
    print("  - data/games/: 5 MP4 + 5 WAV files")

if __name__ == "__main__":
    generate_test_dataset() 

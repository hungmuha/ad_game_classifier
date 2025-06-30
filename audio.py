import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# Configuration
VIDEO_DIRS = ['data/ads', 'data/games']
OUTPUT_AUDIO_FORMAT = 'wav'
CLIP_DURATION = 10  # seconds
SAMPLE_RATE = 16000  # Hz

# Ensure output directories exist
for video_dir in VIDEO_DIRS:
    os.makedirs(video_dir, exist_ok=True)

def extract_audio_from_video(video_path, output_path, duration=CLIP_DURATION):
    try:
        # Load video and extract audio
        video = VideoFileClip(video_path)
        audio = video.audio

        # Write temporary audio file
        temp_audio_path = output_path.replace('.wav', '_temp.wav')
        audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE, verbose=False, logger=None)

        # Load audio and trim/pad to fixed duration
        audio_segment = AudioSegment.from_wav(temp_audio_path)
        target_length_ms = duration * 1000

        if len(audio_segment) > target_length_ms:
            audio_segment = audio_segment[:target_length_ms]
        else:
            silence = AudioSegment.silent(duration=target_length_ms - len(audio_segment))
            audio_segment += silence

        # Export final audio clip
        audio_segment.export(output_path, format=OUTPUT_AUDIO_FORMAT)

        # Clean up
        os.remove(temp_audio_path)
        print(f"Extracted audio: {output_path}")
    except Exception as e:
        print(f"Failed to process {video_path}: {e}")

# Process all videos in the specified directories
for video_dir in VIDEO_DIRS:
    for filename in os.listdir(video_dir):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_audio_path = os.path.join(video_dir, f"{base_name}.wav")
            extract_audio_from_video(video_path, output_audio_path)


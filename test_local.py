import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
from glob import glob
import time
import psutil
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# Import the training components
from cost_optimized_train import (
    CostOptimizedDataset, 
    CostOptimizedClassifier, 
    VIDEO_DIRS, 
    CLIP_DURATION, 
    FPS, 
    NUM_FRAMES, 
    FRAME_SIZE, 
    MFCC_N_MELS, 
    BATCH_SIZE
)

# -------------------------------
# TEST CONFIGURATION
# -------------------------------
TEST_MODE = True  # Enable test mode with smaller parameters
TEST_BATCH_SIZE = 2  # Smaller batch for testing
TEST_EPOCHS = 2  # Just 2 epochs for testing
TEST_SAMPLES = 10  # Test with only 10 samples per class

# -------------------------------
# SYSTEM CHECK
# -------------------------------
def check_system_requirements():
    """Check if the system meets requirements for training."""
    print("🔍 System Requirements Check")
    print("-" * 40)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check CPU and RAM
    cpu_count = psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"CPU Cores: {cpu_count}")
    print(f"RAM: {ram_gb:.1f} GB")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    disk_gb = disk_usage.free / 1e9
    print(f"Free Disk Space: {disk_gb:.1f} GB")
    
    # Recommendations
    print("\n📋 Recommendations:")
    if not cuda_available:
        print("⚠️  No GPU detected. Training will be very slow on CPU.")
    if ram_gb < 8:
        print("⚠️  Less than 8GB RAM. Consider reducing batch size.")
    if disk_gb < 10:
        print("⚠️  Less than 10GB free space. Clean up disk space.")
    
    return cuda_available

# -------------------------------
# DATA VALIDATION
# -------------------------------
def validate_data_structure():
    """Check if data directories and files exist."""
    print("\n📁 Data Structure Validation")
    print("-" * 40)
    
    issues = []
    
    for label, path in VIDEO_DIRS.items():
        print(f"Checking {label} directory: {path}")
        
        # Check if directory exists
        if not os.path.exists(path):
            issues.append(f"Directory not found: {path}")
            continue
        
        # Count video files
        video_path = path+"/video"
        audio_path = path+"/audio"
        video_files = glob(os.path.join(video_path, "*.mp4"))
        print(f"  Found {len(video_files)} MP4 files")
        
        if len(video_files) == 0:
            issues.append(f"No MP4 files found in {path}")
            continue
        
        # Check for corresponding WAV files
        wav_count = 0
        for vf in video_files[:5]:  # Check first 5 files
            base = os.path.splitext(os.path.basename(vf))[0]
            wav_path = os.path.join(audio_path, base + ".wav")
            if os.path.exists(wav_path):
                wav_count += 1
        
        print(f"  {wav_count}/5 have corresponding WAV files")
        
        if wav_count == 0:
            issues.append(f"No WAV files found in {path}")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ Data structure looks good!")
        return True

# -------------------------------
# SAMPLE DATA TEST
# -------------------------------
def test_sample_loading():
    """Test loading a few samples to ensure data pipeline works."""
    print("\n🧪 Sample Data Loading Test")
    print("-" * 40)
    
    try:
        # Create a small test dataset
        dataset = CostOptimizedDataset(VIDEO_DIRS)
        print(f"Dataset size: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("❌ No samples found in dataset")
            return False
        
        # Test loading first few samples
        for i in range(min(3, len(dataset))):
            print(f"\nTesting sample {i+1}:")
            
            start_time = time.time()
            frames, mfcc, label = dataset[i]
            load_time = time.time() - start_time
            
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Video shape: {frames.shape}")
            print(f"  Audio shape: {mfcc.shape}")
            print(f"  Label: {label.item()}")
            
            # Check data types and ranges
            if frames.dtype != torch.float32:
                print(f"  ⚠️  Video dtype: {frames.dtype} (expected float32)")
            if mfcc.dtype != torch.float32:
                print(f"  ⚠️  Audio dtype: {mfcc.dtype} (expected float32)")
            if frames.min() < 0 or frames.max() > 1:
                print(f"  ⚠️  Video range: [{frames.min():.3f}, {frames.max():.3f}] (expected [0,1])")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading samples: {e}")
        return False

# -------------------------------
# MODEL TEST
# -------------------------------
def test_model_forward_pass():
    """Test if the model can process data without errors."""
    print("\n🧠 Model Forward Pass Test")
    print("-" * 40)
    
    try:
        # Create model
        model = CostOptimizedClassifier()
        print(f"Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test with dummy data
        batch_size = 2
        dummy_video = torch.randn(batch_size, NUM_FRAMES, 3, FRAME_SIZE[0], FRAME_SIZE[1])
        dummy_audio = torch.randn(batch_size, MFCC_N_MELS)
        
        print(f"Input video shape: {dummy_video.shape}")
        print(f"Input audio shape: {dummy_audio.shape}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_video, dummy_audio)
        forward_time = time.time() - start_time
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Forward pass time: {forward_time:.3f}s")
        
        # Check output is reasonable
        if output.shape != (batch_size,):
            print(f"❌ Unexpected output shape: {output.shape}")
            return False
        
        if output.min() < 0 or output.max() > 1:
            print(f"❌ Output not in [0,1] range: [{output.min():.3f}, {output.max():.3f}]")
            return False
        
        print("✅ Model forward pass successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in model forward pass: {e}")
        return False

# -------------------------------
# MEMORY TEST
# -------------------------------
def test_memory_usage():
    """Test memory usage with actual data."""
    print("\n💾 Memory Usage Test")
    print("-" * 40)
    
    try:
        # Create dataset and dataloader
        dataset = CostOptimizedDataset(VIDEO_DIRS)
        dataloader = DataLoader(
            dataset, 
            batch_size=TEST_BATCH_SIZE, 
            shuffle=True, 
            num_workers=2
        )
        
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CostOptimizedClassifier().to(device)
        
        # Monitor memory during training loop
        print("Testing memory usage with training loop...")
        
        initial_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else psutil.virtual_memory().used
        
        model.train()
        for i, (videos, audios, labels) in enumerate(dataloader):
            if i >= 3:  # Test only first 3 batches
                break
                
            videos, audios, labels = videos.to(device), audios.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(videos, audios)
            loss = nn.BCELoss()(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Check memory
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated()
                print(f"  Batch {i+1}: GPU Memory: {current_memory / 1e6:.1f} MB")
            else:
                current_memory = psutil.virtual_memory().used
                print(f"  Batch {i+1}: RAM: {(current_memory - initial_memory) / 1e6:.1f} MB")
            
            # Clear gradients
            model.zero_grad()
        
        # Clean up
        del model, dataloader, dataset
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Memory test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in memory test: {e}")
        return False

# -------------------------------
# MINI TRAINING TEST
# -------------------------------
def test_mini_training():
    """Run a mini training session to test the full pipeline."""
    print("\n🚀 Mini Training Test")
    print("-" * 40)
    
    try:
        # Create a small subset of data
        dataset = CostOptimizedDataset(VIDEO_DIRS)
        
        # Limit dataset size for testing
        if len(dataset) > TEST_SAMPLES * 2:
            # Create a subset
            indices = torch.randperm(len(dataset))[:TEST_SAMPLES * 2].tolist()
            dataset = Subset(dataset, indices)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=TEST_BATCH_SIZE, 
            shuffle=True, 
            num_workers=2
        )
        
        # Create model and training components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CostOptimizedClassifier().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        
        print(f"Training on {len(dataset)} samples")
        print(f"Device: {device}")
        
        # Mini training loop
        model.train()
        for epoch in range(TEST_EPOCHS):
            running_loss = 0.0
            batch_count = 0
            
            for videos, audios, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{TEST_EPOCHS}"):
                videos, audios, labels = videos.to(device), audios.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(videos, audios)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
            
            avg_loss = running_loss / batch_count
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        print("✅ Mini training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in mini training: {e}")
        return False

# -------------------------------
# MAIN TEST FUNCTION
# -------------------------------
def run_all_tests():
    """Run all tests and provide a summary."""
    print("🧪 Local Testing Suite for Ad/Game Classifier")
    print("=" * 60)
    
    tests = [
        ("System Requirements", check_system_requirements),
        ("Data Structure", validate_data_structure),
        ("Sample Loading", test_sample_loading),
        ("Model Forward Pass", test_model_forward_pass),
        ("Memory Usage", test_memory_usage),
        ("Mini Training", test_mini_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your script is ready for cloud deployment.")
        print("\n📋 Next steps:")
        print("1. Upload your data to cloud storage")
        print("2. Launch a GPU instance (recommend g4dn.xlarge for cost efficiency)")
        print("3. Install dependencies and run the training script")
        print("4. Monitor training progress and costs")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before cloud deployment.")
        print("\n🔧 Common fixes:")
        print("- Ensure data directories exist and contain MP4/WAV files")
        print("- Check GPU drivers if CUDA tests fail")
        print("- Increase system RAM if memory tests fail")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests() 

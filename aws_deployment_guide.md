# AWS Cloud Deployment Guide for Ad/Game Classifier

## 📋 Pre-Deployment Checklist

### 1. Data Preparation
- [ ] Ensure all MP4 files are in `data/ads/` and `data/games/` directories
- [ ] Ensure corresponding WAV files exist for each MP4
- [ ] Verify data structure matches expected format
- [ ] Calculate total data size (should be ~200GB for 200 hours)

### 2. Code Preparation
- [ ] All training scripts are ready (`cost_optimized_train.py` and `s3_streaming_train.py`)
- [ ] `run_training.sh` script is created and tested locally
- [ ] Dependencies are documented (`Pipfile` or `requirements.txt`)
- [ ] Local testing completed successfully

## 🗂️ Step 1: Data Upload to S3

### S3 Bucket Structure
Your S3 bucket will contain both input data and output results:
```
s3://your-bucket-name/
├── data/                    # Training data (input)
│   ├── ads/
│   │   ├── video/          # MP4 files
│   │   └── audio/          # WAV files
│   └── games/
│       ├── video/          # MP4 files
│       └── audio/          # WAV files
└── results/                # Training results (output)
    ├── 20241201_143022/    # Timestamped results
    │   ├── s3_streaming_best.pth
    │   ├── s3_streaming_final.pth
    │   ├── training_20241201_143022.log
    │   ├── gpu_monitor_20241201_143022.log
    │   └── *.png           # Any plots
    └── 20241201_155630/    # Another training run
        └── ...
```

### Upload Training Data to S3:
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Create S3 bucket
aws s3 mb s3://your-ad-game-data-bucket

# Upload training data (this will take several hours for 200GB)
aws s3 sync data/ s3://your-ad-game-data-bucket/data/ --progress
```

### Alternative Upload Methods:
1. **AWS Console Upload**: Use web interface (slower for large datasets)
2. **AWS DataSync**: For very large datasets
```bash
aws datasync create-task \
  --source-location-arn arn:aws:datasync:region:account:location/loc-xxx \
  --destination-location-arn arn:aws:datasync:region:account:location/loc-yyy \
  --name "AdGameDataTransfer"
```

## 🖥️ Step 2: Launch EC2 Instance

### Recommended Instance: g4dn.xlarge
- **Instance Type**: g4dn.xlarge
- **GPU**: 1x Tesla T4 (16GB VRAM)
- **vCPU**: 4
- **RAM**: 16GB
- **Storage**: 100GB GP3 SSD (reduced from previous recommendations)
- **Cost**: ~$0.526/hour

### Launch Commands:
```bash
# Launch instance with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-001e3e7725d51f534 \
  --instance-type g4dn.xlarge \
  --key-name ad-game-training-key \
  --security-group-ids sg-00ae53cbd052a01de \
  --subnet-id subnet-0403bac7f98bc9edb \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ad-game-training}]'
```

### Alternative: Spot Instances (60-90% cost savings)
```bash
# Request spot instance
aws ec2 request-spot-instances \
  --spot-price "0.20" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

## 🔧 Step 3: Instance Setup

### Connect to Instance:
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

### Download Code:
```bash
# Clone your repository
git clone https://github.com/your-repo/ad_game_classifier.git
cd ad_game_classifier

# Make training script executable
chmod +x run_training.sh
```

## 🎯 Step 4: Training Execution

### Training Approaches

#### **Option A: S3 Streaming (Cost Optimized)**
```bash
# Run with S3 streaming mode (NO data download!)
./run_training.sh s3-streaming your-ad-game-data-bucket

# Or run in background with logging
nohup ./run_training.sh s3-streaming your-ad-game-data-bucket > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

**Pros:**
- ✅ No data transfer costs ($0-5 vs $90-100)
- ✅ Immediate start (no download wait)
- ✅ Works with any dataset size

**Cons:**
- ❌ Slower training (36s/iteration)
- ❌ Network dependent
- ❌ Higher total compute cost

#### **Option B: Local Data (Speed Optimized)**
```bash
# Run training with local data
# auto download data from S3
./run_training.sh aws your-ad-game-data-bucket
```

**Pros:**
- ✅ Much faster training (4-7s/iteration)
- ✅ More reliable (no network dependency)
- ✅ Lower total cost ($111 vs $135-140)

**Cons:**
- ❌ One-time data transfer cost ($90)
- ❌ Requires local storage space

### Enhanced Logging & Monitoring

The updated training scripts now include comprehensive logging and monitoring:

#### **Automatic Logging Features:**
- **Detailed logs**: `logs/training_TIMESTAMP.log`
- **Progress tracking**: `logs/training_TIMESTAMP_progress.json`
- **Model checkpoints**: Saved for each epoch
- **Real-time monitoring**: GPU usage, memory, disk space

#### **Monitoring Commands:**
```bash
# Follow training logs in real-time
tail -f logs/training_*.log

# Check progress summary
python check_progress.py

# Monitor system resources
./monitor_training.sh

# Check GPU usage
watch -n 10 nvidia-smi
```

#### **Progress Tracking Files:**
```
logs/
├── training_20241201_143022.log          # Detailed training log
├── training_20241201_143022_progress.json # Progress summary
└── gpu_monitor_20241201_143022.log       # GPU usage log
```

### What the Script Does Automatically:
1. **System Validation**: Checks GPU, memory, disk space
2. **Dependency Installation**: Installs PyTorch, OpenCV, librosa, boto3, etc.
3. **Data Handling**: Downloads data (AWS mode) or streams from S3 (S3 streaming mode)
4. **Data Validation**: Ensures proper data structure
5. **Resource Monitoring**: Tracks GPU usage and system resources
6. **Enhanced Logging**: Comprehensive progress tracking and logging
7. **Results Management**: Saves models, logs, and plots with timestamps
8. **S3 Results Upload**: Uploads results to `s3://your-bucket/results/TIMESTAMP/`

## 📊 Step 5: Monitoring & Cost Control

### Built-in Monitoring:
The enhanced training scripts now provide:
- **GPU Monitoring**: `nvidia-smi` logging to `gpu_monitor_TIMESTAMP.log`
- **System Resources**: CPU, memory, and disk space tracking
- **Training Progress**: Detailed logs with timestamps and JSON progress files
- **Model Checkpoints**: Automatic saving of best models per epoch
- **Real-time Logging**: Comprehensive iteration-level logging

### Monitoring Scripts:

#### **`monitor_training.sh`** - Quick Status Check:
```bash
#!/bin/bash
echo "=== Training Status ==="
echo "Latest log file:"
ls -t logs/training_*.log | head -1 | xargs tail -20

echo ""
echo "=== Progress Summary ==="
echo "Latest progress file:"
ls -t logs/training_*_progress.json | head -1 | xargs cat | python -m json.tool

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "=== System Resources ==="
echo "Memory:"
free -h
echo "Disk:"
df -h .
```

#### **`check_progress.py`** - Detailed Progress Report:
```python
#!/usr/bin/env python3
import json
import glob
import os
from datetime import datetime

def check_progress():
    # Find latest progress file
    progress_files = glob.glob('logs/training_*_progress.json')
    if not progress_files:
        print("No progress files found!")
        return
    
    latest_file = max(progress_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f"=== Training Progress Report ===")
    print(f"File: {latest_file}")
    print(f"Start time: {data['start_time']}")
    print(f"Best loss: {data['best_loss']:.4f}")
    print(f"Epochs completed: {len(data['epochs'])}")
    
    if data['epochs']:
        latest_epoch = data['epochs'][-1]
        print(f"Latest epoch: {latest_epoch['epoch']}")
        print(f"Latest loss: {latest_epoch['avg_loss']:.4f}")
        print(f"Latest learning rate: {latest_epoch['learning_rate']:.6f}")
        print(f"Latest epoch time: {latest_epoch['time']:.2f}s")
    
    print(f"\n=== Epoch History ===")
    for epoch in data['epochs']:
        print(f"Epoch {epoch['epoch']}: Loss={epoch['avg_loss']:.4f}, "
              f"LR={epoch['learning_rate']:.6f}, Time={epoch['time']:.2f}s")

if __name__ == "__main__":
    check_progress()
```

### Additional Monitoring (Optional):
```bash
# Check GPU usage in real-time
watch -n 1 nvidia-smi

# Check system resources
htop

# Monitor costs (set up CloudWatch alarms)
aws cloudwatch put-metric-alarm \
  --alarm-name "TrainingCostAlarm" \
  --alarm-description "Alert when training costs exceed threshold" \
  --metric-name "EstimatedCharges" \
  --namespace "AWS/Billing" \
  --statistic "Maximum" \
  --period 300 \
  --threshold 10 \
  --comparison-operator "GreaterThanThreshold"
```

## 💾 Step 6: Model & Results Management

### Automatic Results Management:
The enhanced training scripts automatically:
- Creates timestamped results directory (`results_TIMESTAMP/`)
- Saves best and final models (`cost_optimized_best.pth`, `cost_optimized_final.pth`)
- Copies training logs and GPU monitoring data
- Saves progress tracking JSON files
- Saves any generated plots or visualizations
- Uploads everything to S3 at `s3://your-bucket/results/TIMESTAMP/`

### Results Structure:
```
results_20241201_143022/
├── cost_optimized_best_epoch_1.pth      # Best model from epoch 1
├── cost_optimized_best_epoch_2.pth      # Best model from epoch 2
├── cost_optimized_final.pth             # Final model
├── training_20241201_143022.log         # Training logs
├── training_20241201_143022_progress.json # Progress tracking
├── gpu_monitor_20241201_143022.log      # GPU usage logs
└── *.png                                # Any generated plots
```

### Download Results Locally:
```bash
# Download results to your local machine
aws s3 sync s3://your-ad-game-data-bucket/results/ ./downloaded_results/

# Download specific training run
aws s3 sync s3://your-ad-game-data-bucket/results/20241201_143022/ ./specific_run/
```

## 🔄 Step 7: Optimization & Scaling

### Multi-GPU Training (If Needed):
```python
# Modify cost_optimized_train.py for multi-GPU
import torch.nn.parallel
import torch.distributed as dist

# Wrap model for multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Distributed Training (For Large Datasets):
```bash
# Launch multiple instances
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.2xlarge \
  --count 2 \
  --key-name your-key-pair
```

## 🛡️ Step 8: Security & Best Practices

### Security Groups:
```bash
# Create security group for training
aws ec2 create-security-group \
  --group-name "training-sg" \
  --description "Security group for ML training"

# Allow SSH access
aws ec2 authorize-security-group-ingress \
  --group-name "training-sg" \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
```

### IAM Roles:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-ad-game-data-bucket",
        "arn:aws:s3:::your-ad-game-data-bucket/*"
      ]
    }
  ]
}
```

## 💰 Cost Estimation & Optimization

### Cost Comparison:

| Approach | Data Transfer | Compute | Total | Time | Speed |
|----------|---------------|---------|-------|------|-------|
| **S3 Streaming** | $5-10 | $129.40 | **$135-140** | **246 hours** | 36s/iteration |
| **Local Data** | $90 | $21.04 | **$111.12** | **24-40 hours** | 4-7s/iteration |

### Expected Costs (Updated):
- **S3 Streaming**: $135-140 (246 hours)
- **Local Data**: $111.12 (24-40 hours)
- **Spot Instances**: 60-90% savings on compute

### Cost Optimization Tips:
1. **Use Local Data**: Faster and cheaper than S3 streaming
2. **Use Spot Instances**: 60-90% savings
3. **Monitor Progress**: Use enhanced logging to track efficiency
4. **Right-size storage**: Use GP3 instead of GP2
5. **Monitor usage**: Set up CloudWatch alarms
6. **Clean up**: Terminate instances when done

## 🚨 Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Data Loading**: S3 streaming may be slower than local files
3. **GPU Driver Issues**: Use Deep Learning AMI
4. **Network Timeouts**: Use larger instance types
5. **S3 Access Issues**: Check IAM permissions and bucket name
6. **Training Stops**: Check auto-stop settings and logs

### Debug Commands:
```bash
# Check GPU status
nvidia-smi

# Check disk space
df -h

# Check memory usage
free -h

# Check network
ping -c 3 s3.amazonaws.com

# Check training logs
tail -f logs/training_*.log

# Check progress
python check_progress.py

# Test S3 access
aws s3 ls s3://your-bucket-name/

# Test S3 streaming
python -c "import boto3; print('boto3 available')"
```

### Script-Specific Troubleshooting:
```bash
# Test script locally first
./run_training.sh local

# Test S3 streaming mode
./run_training.sh s3-streaming your-bucket-name

# Test local data mode
./run_training.sh aws your-bucket-name

# Check script permissions
ls -la run_training.sh

# Check Python environment
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Performance Monitoring

### Training Metrics to Track:
- Loss per epoch
- GPU utilization
- Memory usage
- Training time per epoch
- Cost per epoch
- Learning rate progression

### Set Up Monitoring:
```bash
# Install monitoring tools
pip install tensorboard wandb

# Start TensorBoard
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# Monitor training progress
watch -n 300 python check_progress.py
```

## 🚀 Quick Start Summary

1. **Prepare Locally**: Test `./run_training.sh local`
2. **Upload Data**: `aws s3 sync data/ s3://your-bucket/data/`
3. **Launch EC2**: Use g4dn.xlarge with Deep Learning AMI
4. **Download Code**: `git clone` your repository
5. **Choose Approach**:
   - **Fast**: `./run_training.sh aws your-bucket-name` (download data first)
   - **Cost-optimized**: `./run_training.sh s3-streaming your-bucket-name`
6. **Monitor**: Use enhanced logging and monitoring scripts
7. **Download Results**: `aws s3 sync s3://your-bucket/results/ ./results/`

## 📝 Complete Example Workflow

### **Option A: Local Data (Recommended)**
```bash
# 1. Upload training data to S3 (one-time setup)
aws s3 sync data/ s3://my-ml-bucket/data/

# 2. Launch EC2 instance and connect
ssh -i key.pem ubuntu@your-instance-ip

# 3. Clone repository and download data
git clone https://github.com/your-repo/ad_game_classifier.git
cd ad_game_classifier
chmod +x run_training.sh
aws s3 sync s3://my-ml-bucket/data/ data/

# 4. Run training with enhanced logging
./run_training.sh aws my-ml-bucket

# 5. Monitor progress
tail -f logs/training_*.log
python check_progress.py

# 6. Download results locally
aws s3 sync s3://my-ml-bucket/results/ ./downloaded_results/
```

### **Option B: S3 Streaming**
```bash
# 1. Upload training data to S3 (one-time setup)
aws s3 sync data/ s3://my-ml-bucket/data/

# 2. Launch EC2 instance and connect
ssh -i key.pem ubuntu@your-instance-ip

# 3. Clone repository and run training
git clone https://github.com/your-repo/ad_game_classifier.git
cd ad_game_classifier
chmod +x run_training.sh
./run_training.sh s3-streaming my-ml-bucket

# 4. Monitor progress
tail -f logs/training_*.log
python check_progress.py

# 5. Download results locally
aws s3 sync s3://my-ml-bucket/results/ ./downloaded_results/
```

The enhanced training scripts now provide comprehensive logging, monitoring, and progress tracking, making it much easier to track training progress and optimize performance!

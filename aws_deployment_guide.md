# AWS Cloud Deployment Guide for Ad/Game Classifier

## 📋 Pre-Deployment Checklist

### 1. Data Preparation
- [ ] Ensure all MP4 files are in `data/ads/` and `data/games/` directories
- [ ] Ensure corresponding WAV files exist for each MP4
- [ ] Verify data structure matches expected format
- [ ] Calculate total data size (should be ~200GB for 200 hours)

### 2. Code Preparation
- [ ] All training scripts are ready (`cost_optimized_train.py`)
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
    │   ├── cost_optimized_best.pth
    │   ├── cost_optimized_final.pth
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
- **Storage**: 100GB GP3 SSD
- **Cost**: ~$0.526/hour

### Launch Commands:
```bash
# Launch instance with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
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

### The `run_training.sh` Script
The repository now includes a comprehensive `run_training.sh` script that handles:
- ✅ Environment setup and validation
- ✅ Dependency installation (on AWS mode)
- ✅ **S3 data download** (NEW!)
- ✅ Data structure validation
- ✅ GPU and system resource monitoring
- ✅ Automatic results management
- ✅ S3 results upload
- ✅ Cost control with auto-stop

### Run Training on AWS:
```bash
# Run with AWS mode and S3 bucket (downloads data automatically)
./run_training.sh aws your-ad-game-data-bucket

# Or run in background with logging
nohup ./run_training.sh aws your-ad-game-data-bucket > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### What the Script Does Automatically:
1. **System Validation**: Checks GPU, memory, disk space
2. **Dependency Installation**: Installs PyTorch, OpenCV, librosa, etc.
3. **S3 Data Download**: Downloads training data from `s3://your-bucket/data/` to local `data/` directory
4. **Data Validation**: Ensures proper data structure
5. **Resource Monitoring**: Tracks GPU usage and system resources
6. **Auto-Stop Setup**: Prevents runaway costs (20-hour limit)
7. **Results Management**: Saves models, logs, and plots with timestamps
8. **S3 Results Upload**: Uploads results to `s3://your-bucket/results/TIMESTAMP/`

### S3 Data Download Features:
- **Smart Download**: Only downloads if data doesn't already exist locally
- **Progress Tracking**: Shows download progress for large datasets
- **Error Handling**: Validates S3 bucket access and data existence
- **Verification**: Confirms download completion and counts files

## 📊 Step 5: Monitoring & Cost Control

### Built-in Monitoring:
The `run_training.sh` script automatically provides:
- **GPU Monitoring**: `nvidia-smi` logging to `gpu_monitor_TIMESTAMP.log`
- **System Resources**: CPU, memory, and disk space tracking
- **Training Progress**: Detailed logs with timestamps
- **Cost Control**: Auto-stop after 20 hours

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
The `run_training.sh` script automatically:
- Creates timestamped results directory (`results_TIMESTAMP/`)
- Saves best and final models
- Copies training logs and GPU monitoring data
- Saves any generated plots or visualizations
- Uploads everything to S3 at `s3://your-bucket/results/TIMESTAMP/`

### Results Structure:
```
results_20241201_143022/
├── cost_optimized_best.pth      # Best model during training
├── cost_optimized_final.pth     # Final model
├── training_20241201_143022.log # Training logs
├── gpu_monitor_20241201_143022.log # GPU usage logs
└── *.png                        # Any generated plots
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

### Expected Costs:
- **g4dn.xlarge**: $0.526/hour
- **20 hours training**: ~$10.52
- **Data transfer**: ~$2-5
- **Storage**: ~$2/month
- **Total**: ~$15-20

### Cost Optimization Tips:
1. **Use Spot Instances**: 60-90% savings
2. **Auto-stop**: Built into `run_training.sh` (20-hour limit)
3. **Right-size storage**: Use GP3 instead of GP2
4. **Monitor usage**: Set up CloudWatch alarms
5. **Clean up**: Terminate instances when done

## 🚨 Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Data Loading**: Use more workers or SSD storage
3. **GPU Driver Issues**: Use Deep Learning AMI
4. **Network Timeouts**: Use larger instance types
5. **S3 Access Issues**: Check IAM permissions and bucket name

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
tail -f training.log

# Check script logs
tail -f training_TIMESTAMP.log

# Test S3 access
aws s3 ls s3://your-bucket-name/

# Check downloaded data
ls -la data/ads/video/ | head -5
ls -la data/games/video/ | head -5
```

### Script-Specific Troubleshooting:
```bash
# Test script locally first
./run_training.sh local

# Check script permissions
ls -la run_training.sh

# Validate data structure
find data/ -name "*.mp4" | head -5
find data/ -name "*.wav" | head -5

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

### Set Up Monitoring:
```bash
# Install monitoring tools
pip install tensorboard wandb

# Start TensorBoard
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

## 🚀 Quick Start Summary

1. **Prepare Locally**: Test `./run_training.sh local`
2. **Upload Data**: `aws s3 sync data/ s3://your-bucket/data/`
3. **Launch EC2**: Use g4dn.xlarge with Deep Learning AMI
4. **Download Code**: `git clone` your repository
5. **Run Training**: `./run_training.sh aws your-bucket-name`
6. **Monitor**: Check logs and S3 for results
7. **Download Results**: `aws s3 sync s3://your-bucket/results/ ./results/`

## 📝 Complete Example Workflow

```bash
# 1. Upload training data to S3 (one-time setup)
aws s3 sync data/ s3://my-ml-bucket/data/

# 2. Launch EC2 instance and connect
ssh -i key.pem ubuntu@your-instance-ip

# 3. Clone repository and run training
git clone https://github.com/your-repo/ad_game_classifier.git
cd ad_game_classifier
chmod +x run_training.sh
./run_training.sh aws my-ml-bucket

# 4. Download results locally (optional)
aws s3 sync s3://my-ml-bucket/results/ ./downloaded_results/
```

The `run_training.sh` script now handles the complete workflow from S3 data download to results upload, making AWS deployment much simpler!

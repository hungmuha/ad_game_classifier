# AWS Cloud Deployment Guide for Ad/Game Classifier

## 📋 Pre-Deployment Checklist

### 1. Data Preparation
- [ ] Ensure all MP4 files are in `data/ads/` and `data/games/` directories
- [ ] Ensure corresponding WAV files exist for each MP4
- [ ] Verify data structure matches expected format
- [ ] Calculate total data size (should be ~200GB for 200 hours)

### 2. Code Preparation
- [ ] All training scripts are ready (`cost_optimized_train.py`)
- [ ] Dependencies are documented (`requirements_test.txt`)
- [ ] Local testing completed successfully

## 🗂️ Step 1: Data Upload to S3

### Option A: AWS CLI Upload (Recommended)
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Create S3 bucket
aws s3 mb s3://your-ad-game-data-bucket

# Upload data (this will take several hours for 200GB)
aws s3 sync data/ s3://your-ad-game-data-bucket/data/ --progress
```

### Option B: AWS Console Upload
1. Go to S3 Console
2. Create new bucket
3. Upload data folder through web interface
4. Note: Slower for large datasets

### Option C: AWS DataSync (For Very Large Datasets)
```bash
# Create DataSync task for efficient transfer
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

### Install Dependencies:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python librosa numpy tqdm psutil moviepy scikit-learn matplotlib

# Install system dependencies
sudo apt install -y ffmpeg libsndfile1
```

### Download Code and Data:
```bash
# Clone your repository or upload code
git clone https://github.com/your-repo/ads-nfl-model.git
cd ads-nfl-model/ad_game_classifier

# Download data from S3
aws s3 sync s3://your-ad-game-data-bucket/data/ data/
```

## 🎯 Step 4: Training Execution

### Create Training Script:
```bash
# Create a training launcher script
cat > run_training.sh << 'EOF'
#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Monitor GPU usage
nvidia-smi -l 1 &

# Run training
python cost_optimized_train.py

# Save logs
cp *.pth /tmp/  # Backup models
EOF

chmod +x run_training.sh
```

### Run Training:
```bash
# Start training
./run_training.sh

# Or run in background with logging
nohup ./run_training.sh > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## 📊 Step 5: Monitoring & Cost Control

### Monitor Training:
```bash
# Check GPU usage
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

### Set Up Auto-Stop (Cost Control):
```bash
# Create auto-stop script
cat > auto_stop.sh << 'EOF'
#!/bin/bash

# Stop instance after 20 hours (safety measure)
sleep 72000  # 20 hours
aws ec2 stop-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
EOF

chmod +x auto_stop.sh
nohup ./auto_stop.sh &
```

## 💾 Step 6: Model & Results Management

### Save Results to S3:
```bash
# Create results backup script
cat > save_results.sh << 'EOF'
#!/bin/bash

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Save models and logs
aws s3 cp *.pth s3://your-ad-game-data-bucket/models/$TIMESTAMP/
aws s3 cp training.log s3://your-ad-game-data-bucket/logs/$TIMESTAMP/
aws s3 cp *.png s3://your-ad-game-data-bucket/plots/$TIMESTAMP/ 2>/dev/null || true

echo "Results saved to S3 with timestamp: $TIMESTAMP"
EOF

chmod +x save_results.sh
```

### Download Results Locally:
```bash
# Download from your local machine
aws s3 sync s3://your-ad-game-data-bucket/models/ ./downloaded_models/
aws s3 sync s3://your-ad-game-data-bucket/logs/ ./downloaded_logs/
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
2. **Auto-stop**: Prevent runaway costs
3. **Right-size storage**: Use GP3 instead of GP2
4. **Monitor usage**: Set up CloudWatch alarms
5. **Clean up**: Terminate instances when done

## 🚨 Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or use smaller model
2. **Slow Data Loading**: Use more workers or SSD storage
3. **GPU Driver Issues**: Use Deep Learning AMI
4. **Network Timeouts**: Use larger instance types

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

This guide covers the complete AWS deployment process for your cost-optimized training script! 

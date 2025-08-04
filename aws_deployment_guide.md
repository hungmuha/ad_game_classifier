# AWS Cloud Deployment Guide for Ad/Game Classifier

## 📋 Pre-Deployment Checklist

### 1. Data Preparation
- [ ] Ensure all MP4 files are in `data/ads/` and `data/games/` directories
- [ ] Ensure corresponding WAV files exist for each MP4
- [ ] Verify data structure matches expected format
- [ ] Calculate total data size (should be ~200GB for 200 hours)
- [ ] **NEW**: Prepare validation data in `data/validation/` directory (see Validation Setup section)

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
│   ├── games/
│   │   ├── video/          # MP4 files
│   │   └── audio/          # WAV files
│   └── validation/         # NEW: Validation data (unseen by model)
│       ├── ads/
│       │   ├── video/      # MP4 files (unseen)
│       │   └── audio/      # WAV files (unseen)
│       └── games/
│           ├── video/      # MP4 files (unseen)
│           └── audio/      # WAV files (unseen)
└── results/                # Training results (output)
    ├── 20241201_143022/    # Timestamped results
    │   ├── training/       # Training logs
    │   ├── monitoring/     # GPU/system monitoring
    │   ├── system/         # System info logs
    │   ├── results/        # Model files and plots
    │   └── session_summary.txt
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
aws s3 sync data/ s3://your-ad-game-data-bucket/data/
```

## 🔍 Step 1.5: Validation Data Setup

### Why Validation Data is Needed
Since your model has already seen all the training data in epoch 1, we need **separate validation data** to properly evaluate the model's performance and prevent overfitting.

### Local Validation Data Structure
```
data/
├── ads/           (Training data - already seen by model)
├── games/         (Training data - already seen by model)  
└── validation/    (NEW - unseen validation data)
    ├── ads/
    │   ├── video/ (MP4 files - unseen)
    │   └── audio/ (WAV files - unseen)
    └── games/
        ├── video/ (MP4 files - unseen)
        └── audio/ (WAV files - unseen)
```

### Setup Validation Data Locally:
```bash
# 1. Check current validation data status
python setup_validation_data.py --check

# 2. Create validation directory structure
python setup_validation_data.py --create

# 3. Add validation files to the new directories:
#    - data/validation/ads/video/ (ad video files)
#    - data/validation/ads/audio/ (ad audio files)
#    - data/validation/games/video/ (game video files)
#    - data/validation/games/audio/ (game audio files)

# 4. Upload validation data to S3
python setup_validation_data.py --upload your-ad-game-data-bucket
```

### Validation Data Requirements:
- **Size**: 10-20% of your total dataset
- **Balance**: Equal number of ads and games
- **Quality**: Similar to training data
- **Unseen**: Completely separate from training data

### Alternative: Upload Validation Data Directly
```bash
# Upload validation data to S3
aws s3 sync data/validation/ s3://your-ad-game-data-bucket/data/validation/
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
chmod +x run_training.sh check_progress.py
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

The updated training scripts now include comprehensive logging and monitoring with organized directory structure:

#### **New Organized Logging Structure:**
```
logging/
└── session_20241201_143022/
    ├── training/
    │   └── training_20241201_143022.log
    ├── monitoring/
    │   ├── gpu_monitor_20241201_143022.log
    │   ├── training_monitor_20241201_143022.log
    │   └── progress_checker_20241201_143022.log
    ├── system/
    │   └── system_20241201_143022.log
    ├── results/
    │   ├── cost_optimized_best.pth
    │   ├── cost_optimized_final.pth
    │   └── *.png
    └── session_summary.txt
```

#### **Automatic Logging Features:**
- **Organized logs**: All logs organized by type in timestamped sessions
- **Training logs**: Detailed iteration-level training progress
- **GPU monitoring**: Real-time GPU usage and temperature tracking
- **System monitoring**: CPU, memory, and disk usage
- **Progress tracking**: JSON-based progress summaries
- **Session summaries**: Complete session overview with file locations

## 📊 Step 5: Monitoring & Progress Tracking

### **Enhanced Monitoring System**

The new monitoring system provides comprehensive tracking with organized logging:

#### **1. Real-Time Monitoring Commands:**

```bash
# Start monitoring before training (in separate terminal)
nohup nvidia-smi -l 1 > logging/session_*/monitoring/gpu_monitor_*.log &
nohup ./check_progress.py --continuous &

# Start training
./run_training.sh aws your-bucket-name
```

#### **2. Check Training Status:**

```bash
# Check if training is running
ps aux | grep -E "(python.*train|s3_streaming_train|cost_optimized_train)"

# Check GPU usage
nvidia-smi

# Check latest session
ls -la logging/session_*/

# Check training logs
tail -f logging/session_*/training/training_*.log

# Check GPU monitoring
tail -f logging/session_*/monitoring/gpu_monitor_*.log
```

#### **3. Progress Tracking Scripts:**

**`check_progress.py` - Comprehensive Status Check:**
```bash
# Single status check
python check_progress.py

# Continuous monitoring
python check_progress.py --continuous

# Check specific session
python check_progress.py --session session_20241201_143022
```

**Real-Time Monitoring:**
```bash
# Monitor training logs in real-time
tail -f logging/session_*/training/*.log

# Monitor GPU usage
watch -n 10 nvidia-smi
```

#### **4. Monitoring Features:**

**Training Process Monitoring:**
- ✅ **Process tracking**: Monitors training process PID
- ✅ **GPU utilization**: Real-time GPU usage and memory
- ✅ **System resources**: CPU, memory, and disk usage
- ✅ **Training progress**: Latest training output and logs
- ✅ **Error detection**: Automatic error detection and logging

**Logging Organization:**
- ✅ **Session-based**: Each training run gets its own session directory
- ✅ **Categorized logs**: Training, monitoring, system, and results separated
- ✅ **Timestamped**: All files include timestamps for easy tracking
- ✅ **Session summaries**: Complete overview of each training session

#### **5. Monitoring Commands Summary:**

```bash
# Quick status check
python check_progress.py

# Continuous monitoring
python check_progress.py --continuous

# Monitor training process
tail -f logging/session_*/training/*.log

# Check GPU usage
watch -n 10 nvidia-smi

# Check all logs in current session
tail -f logging/session_*/*.log

# Check specific log types
tail -f logging/session_*/training/*.log
tail -f logging/session_*/monitoring/*.log
tail -f logging/session_*/system/*.log

# Check session summary
cat logging/session_*/session_summary.txt
```

#### **6. Monitoring Dashboard:**

```bash
# Create a monitoring dashboard
echo "=== TRAINING MONITORING DASHBOARD ==="
echo "Session: $(ls -t logging/session_*/ | head -1)"
echo "Training Status: $(ps aux | grep -E "(python.*train)" | grep -v grep | wc -l) processes running"
echo "GPU Usage: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
echo "Latest Log: $(ls -t logging/session_*/training/*.log | head -1)"
echo "====================================="
```

### **What the Enhanced Scripts Do Automatically:**

1. **Organized Logging**: Creates structured logging directory with categorized logs
2. **Session Management**: Each training run gets a unique session with timestamp
3. **Real-Time Monitoring**: GPU, system resources, and training progress
4. **Progress Tracking**: Detailed logs with timestamps and JSON summaries
5. **Error Detection**: Automatic error logging and recovery suggestions
6. **Results Management**: Organized model files, plots, and session summaries
7. **S3 Integration**: Automatic upload of complete session data to S3

## 💾 Step 6: Model & Results Management

### **Enhanced Results Management:**

The updated training scripts automatically create organized results:

#### **Results Structure:**
```
logging/session_20241201_143022/
├── training/
│   └── training_20241201_143022.log
├── monitoring/
│   ├── gpu_monitor_20241201_143022.log
│   ├── training_monitor_20241201_143022.log
│   └── progress_checker_20241201_143022.log
├── system/
│   └── system_20241201_143022.log
├── results/
│   ├── cost_optimized_best.pth
│   ├── cost_optimized_final.pth
│   └── *.png
└── session_summary.txt
```

#### **Automatic Features:**
- ✅ **Session-based organization**: Each training run in its own directory
- ✅ **Complete logging**: All logs, models, and monitoring data preserved
- ✅ **Session summaries**: Overview of each training session
- ✅ **S3 upload**: Complete session data uploaded to S3
- ✅ **Timestamped files**: All files include timestamps for tracking

### **Download Results Locally:**
```bash
# Download complete session results
aws s3 sync s3://your-ad-game-data-bucket/results/session_20241201_143022/ ./downloaded_session/

# Download specific session
aws s3 sync s3://your-ad-game-data-bucket/results/session_20241201_143022/ ./specific_session/

# Download all results
aws s3 sync s3://your-ad-game-data-bucket/results/ ./all_results/
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
tail -f logging/session_*/training/*.log

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
ls -la run_training.sh check_progress.py

# Check Python environment
python -c "import torch; print(torch.cuda.is_available())"

# Check monitoring scripts
python check_progress.py
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
tensorboard --logdir=./logging --host=0.0.0.0 --port=6006

# Monitor training progress
watch -n 300 python check_progress.py

# Monitor specific session
python check_progress.py --session session_20241201_143022
```

## 🚀 Quick Start Summary

1. **Prepare Locally**: Test `./run_training.sh local`
2. **Upload Data**: `aws s3 sync data/ s3://your-bucket/data/`
3. **Launch EC2**: Use g4dn.xlarge with Deep Learning AMI
4. **Download Code**: `git clone` your repository
5. **Setup Monitoring**: Start monitoring scripts in separate terminals
6. **Choose Approach**:
   - **Fast**: `./run_training.sh aws your-bucket-name` (download data first)
   - **Cost-optimized**: `./run_training.sh s3-streaming your-bucket-name`
7. **Monitor**: Use enhanced logging and monitoring scripts
8. **Download Results**: `aws s3 sync s3://your-bucket/results/ ./results/`

## 🎯 Step 4: Updated Training with Validation

### **New Training Approach**
The training now includes **validation capabilities** to prevent overfitting and provide unbiased model evaluation.

### **Option A: Validation-Enhanced Training (Recommended)**
```bash
# Run training with validation (automatically detects checkpoint)
./run_training.sh aws your-ad-game-data-bucket
```

**Features:**
- ✅ **Automatic checkpoint detection** (resumes from epoch 2)
- ✅ **Separate validation data** for unbiased evaluation
- ✅ **Validation-based early stopping** prevents overfitting
- ✅ **S3 upload capability** for results
- ✅ **Single entry point** - everything flows through run_training.sh

### **Option B: S3 Streaming (Cost Optimized)**
```bash
# Run training with S3 streaming (no data download)
./run_training.sh s3-streaming your-ad-game-data-bucket
```

### **Option C: Local Training**
```bash
# Run training locally with validation
./run_training.sh local
```

### **Training Output Example:**
```
Starting cost-optimized training with validation...
Training data: {'ads': 'data/ads', 'games': 'data/games'}
Validation data: {'ads': 'data/validation/ads', 'games': 'data/validation/games'}
Loading training dataset...
Loading validation dataset...
Validation dataset loaded: 150 samples
Train batches: 125, Validation batches: 19

Epoch 1 - Train Loss: 0.6234, Val Loss: 0.5891, Val Acc: 0.6543
Epoch 2 - Train Loss: 0.5123, Val Loss: 0.5234, Val Acc: 0.7234
```

### **Key Benefits of Validation:**
- **Overfitting Detection**: Validation loss shows true generalization
- **Better Early Stopping**: Stops based on validation loss (not training loss)
- **Unbiased Evaluation**: Uses completely unseen data
- **Performance Metrics**: Validation accuracy for model quality assessment

## 📝 Complete Example Workflow

### **Option A: Validation-Enhanced Training (Recommended)**
```bash
# 1. Upload training and validation data to S3 (one-time setup)
aws s3 sync data/ s3://my-ml-bucket/data/

# 2. Launch EC2 instance and connect
ssh -i key.pem ubuntu@your-instance-ip

# 3. Clone repository and setup monitoring
git clone https://github.com/your-repo/ad_game_classifier.git
cd ad_game_classifier
chmod +x run_training.sh check_progress.py

# 4. Install additional dependencies
pip install scikit-learn boto3

# 5. Start monitoring (in separate terminals)
# Terminal 1: GPU monitoring
nohup nvidia-smi -l 1 > logging/session_*/monitoring/gpu_monitor_*.log &

# Terminal 2: Progress checking
nohup python check_progress.py --continuous &

# 6. Run validation-enhanced training (single command)
./run_training.sh aws my-ml-bucket

# 7. Monitor progress
tail -f logging/session_*/training/*.log
python check_progress.py

# 8. Download results locally
aws s3 sync s3://my-ml-bucket/results/ ./downloaded_results/
```

### **Option B: S3 Streaming**
```bash
# 1. Upload training data to S3 (one-time setup)
aws s3 sync data/ s3://my-ml-bucket/data/

# 2. Launch EC2 instance and connect
ssh -i key.pem ubuntu@your-instance-ip

# 3. Clone repository and setup monitoring
git clone https://github.com/your-repo/ad_game_classifier.git
cd ad_game_classifier
chmod +x run_training.sh check_progress.py

# 4. Start monitoring (in separate terminals)
# Terminal 1: GPU monitoring
nohup nvidia-smi -l 1 > logging/session_*/monitoring/gpu_monitor_*.log &

# Terminal 2: Progress checking
nohup python check_progress.py --continuous &

# 5. Run training with enhanced logging
./run_training.sh s3-streaming my-ml-bucket

# 6. Monitor progress
tail -f logging/session_*/training/*.log
python check_progress.py

# 7. Download results locally
aws s3 sync s3://my-ml-bucket/results/ ./downloaded_results/
```

### **Monitoring Dashboard Commands:**
```bash
# Quick status overview
echo "=== TRAINING MONITORING DASHBOARD ==="
echo "Session: $(ls -t logging/session_*/ | head -1)"
echo "Training Status: $(ps aux | grep -E "(python.*train)" | grep -v grep | wc -l) processes running"
echo "GPU Usage: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
echo "Latest Log: $(ls -t logging/session_*/training/*.log | head -1)"
echo "====================================="

# Check session summary
cat logging/session_*/session_summary.txt

# Monitor all logs
tail -f logging/session_*/*.log
```

The enhanced training scripts now provide comprehensive logging, monitoring, and progress tracking with organized directory structure, making it much easier to track training progress and optimize performance!

## 🔧 Validation Troubleshooting

### **No Validation Data Found**
```bash
# Check validation data structure
python setup_validation_data.py --check

# Create validation directories
python setup_validation_data.py --create

# Add validation files manually, then retry training
```

### **Training Fails with Validation**
```bash
# Check dependencies
pip install scikit-learn boto3

# Check data structure
ls -la data/validation/ads/video/
ls -la data/validation/games/video/

# Run with fallback (uses 20% of training data)
python cost_optimized_train.py
```

### **S3 Upload Fails**
```bash
# Check AWS credentials
aws configure list

# Test S3 access
aws s3 ls s3://your-bucket-name/

# Check bucket permissions
aws s3 ls s3://your-bucket-name/data/validation/
```

### **Checkpoint Issues**
```bash
# List available checkpoints
ls -la *.pth

# The run_training.sh script automatically detects and uses checkpoints
./run_training.sh aws your-bucket-name

# For manual checkpoint usage (advanced)
python cost_optimized_train.py --resume cost_optimized_best_epoch_1.pth
```

### **Resuming from Specific Epoch**
```bash
# Resume from a specific checkpoint file
./run_training.sh aws your-bucket-name cost_optimized_checkpoint_epoch_4.pth

# Resume from a safety checkpoint (most recent progress)
./run_training.sh aws your-bucket-name cost_optimized_safety_checkpoint_epoch_4_iter_500.pth

# Resume from a best model file (continues from next epoch)
./run_training.sh aws your-bucket-name cost_optimized_best_epoch_3_val.pth
```

**Checkpoint Priority:**
1. Safety checkpoints (most recent progress)
2. Regular checkpoints (full training state)
3. Best model files (model only, continues from next epoch)

**Examples:**
- `cost_optimized_checkpoint_epoch_4.pth` → Continues from epoch 4
- `cost_optimized_best_epoch_3_val.pth` → Continues from epoch 4 (next after 3)
- `cost_optimized_safety_checkpoint_epoch_4_iter_500.pth` → Continues from epoch 4, iteration 501

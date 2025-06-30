#!/bin/bash

# AWS Setup Script for Ad/Game Classifier Training
# This script sets up an EC2 instance for training

set -e  # Exit on any error

echo "🚀 Setting up AWS environment for ad/game classifier training..."

# -------------------------------
# CONFIGURATION
# -------------------------------
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID="ami-0c02fb55956c7d316"  # Deep Learning AMI (Ubuntu 18.04)
VOLUME_SIZE=100
KEY_NAME="your-key-pair"  # Change this to your key pair name
SECURITY_GROUP="training-sg"
BUCKET_NAME="your-ad-game-data-bucket"  # Change this to your bucket name

# -------------------------------
# CREATE SECURITY GROUP
# -------------------------------
echo "🔒 Creating security group..."

# Check if security group exists
if aws ec2 describe-security-groups --group-names $SECURITY_GROUP 2>/dev/null; then
    echo "Security group $SECURITY_GROUP already exists"
else
    aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "Security group for ML training"
    
    # Allow SSH access
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
    
    # Allow HTTP for TensorBoard (optional)
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 6006 \
        --cidr 0.0.0.0/0
    
    echo "Security group $SECURITY_GROUP created successfully"
fi

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP --query 'SecurityGroups[0].GroupId' --output text)

# -------------------------------
# CREATE S3 BUCKET (if needed)
# -------------------------------
echo "🗂️  Setting up S3 bucket..."

if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    aws s3 mb s3://$BUCKET_NAME
    echo "S3 bucket $BUCKET_NAME created"
else
    echo "S3 bucket $BUCKET_NAME already exists"
fi

# -------------------------------
# LAUNCH EC2 INSTANCE
# -------------------------------
echo "🖥️  Launching EC2 instance..."

# Create user data script for automatic setup
cat > user_data.sh << 'EOF'
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python librosa numpy tqdm psutil moviepy scikit-learn matplotlib

# Install system dependencies
sudo apt install -y ffmpeg libsndfile1 htop

# Create training directory
mkdir -p /home/ubuntu/training
cd /home/ubuntu/training

# Download training code (you'll need to upload this)
# git clone https://github.com/your-repo/ads-nfl-model.git

# Download data from S3
aws s3 sync s3://your-ad-game-data-bucket/data/ data/

# Create training script
cat > run_training.sh << 'TRAINING_EOF'
#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Monitor GPU usage
nvidia-smi -l 1 &

# Run training
python cost_optimized_train.py

# Save results to S3
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
aws s3 cp *.pth s3://your-ad-game-data-bucket/models/$TIMESTAMP/
aws s3 cp training.log s3://your-ad-game-data-bucket/logs/$TIMESTAMP/

echo "Training completed and results saved to S3"
TRAINING_EOF

chmod +x run_training.sh

# Create auto-stop script (safety measure)
cat > auto_stop.sh << 'STOP_EOF'
#!/bin/bash

# Stop instance after 20 hours
sleep 72000
aws ec2 stop-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
STOP_EOF

chmod +x auto_stop.sh
nohup ./auto_stop.sh &

echo "Setup completed successfully!"
EOF

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --user-data file://user_data.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ad-game-training}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched with ID: $INSTANCE_ID"

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Instance is ready!"
echo "📋 Connection details:"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  SSH Command: ssh -i $KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo ""
echo "📊 Next steps:"
echo "1. Upload your training code to the instance"
echo "2. Upload your data to S3: aws s3 sync data/ s3://$BUCKET_NAME/data/"
echo "3. Connect to instance and start training"
echo "4. Monitor progress: tail -f training.log"
echo ""
echo "💰 Estimated cost: ~$0.526/hour"
echo "⏰ Auto-stop set to 20 hours (~$10.52 total)"

# Clean up
rm user_data.sh 

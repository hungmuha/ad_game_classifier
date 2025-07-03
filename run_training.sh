#!/bin/bash

# ========================================
# Ad/Game Classifier Training Script
# ========================================
# This script is designed to run both locally and on AWS
# Usage: ./run_training.sh [local|aws] [s3-bucket-name]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_${TIMESTAMP}.log"

# S3 Configuration
S3_BUCKET_NAME="${2:-your-ad-game-data-bucket}"  # Can be passed as second argument
S3_DATA_PATH="data"

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    else
        log "No GPU detected, will use CPU"
    fi
}

# Function to check system resources
check_resources() {
    log "System resources:"
    log "CPU cores: $(nproc)"
    log "Memory: $(free -h | grep Mem | awk '{print $2}')"
    log "Disk space: $(df -h . | tail -1 | awk '{print $4}') available"
}

# Function to install dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Check if we're in a virtual environment or have pipenv
    if [ -f "Pipfile" ]; then
        log "Using pipenv..."
        pip install pipenv
        pipenv install
    else
        log "Installing packages directly..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install opencv-python librosa numpy tqdm psutil moviepy scikit-learn matplotlib
    fi
    
    # Install system dependencies if on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log "Installing system dependencies..."
        sudo apt update -qq
        sudo apt install -y ffmpeg libsndfile1
    fi
}

# Function to download data from S3
download_data_from_s3() {
    if [ "$1" = "aws" ] && command -v aws &> /dev/null; then
        log "Checking if data needs to be downloaded from S3..."
        
        # Check if data already exists locally
        if [ -d "data/ads/video" ] && [ -d "data/games/video" ]; then
            log "Data directories already exist locally. Skipping S3 download."
            return 0
        fi
        
        log "Downloading data from S3..."
        log "S3 Bucket: $S3_BUCKET_NAME"
        log "S3 Path: s3://$S3_BUCKET_NAME/$S3_DATA_PATH/"
        
        # Check if bucket exists and is accessible
        if ! aws s3 ls "s3://$S3_BUCKET_NAME" >/dev/null 2>&1; then
            log "ERROR: Cannot access S3 bucket '$S3_BUCKET_NAME'"
            log "Please check:"
            log "  1. AWS credentials are configured"
            log "  2. Bucket name is correct"
            log "  3. IAM permissions allow S3 access"
            exit 1
        fi
        
        # Check if data exists in S3
        if ! aws s3 ls "s3://$S3_BUCKET_NAME/$S3_DATA_PATH/" >/dev/null 2>&1; then
            log "ERROR: Data not found in s3://$S3_BUCKET_NAME/$S3_DATA_PATH/"
            log "Please upload your data to S3 first:"
            log "  aws s3 sync data/ s3://$S3_BUCKET_NAME/$S3_DATA_PATH/"
            exit 1
        fi
        
        # Download data with progress
        log "Starting data download (this may take several hours for large datasets)..."
        aws s3 sync "s3://$S3_BUCKET_NAME/$S3_DATA_PATH/" "data/" --progress
        
        # Verify download
        if [ -d "data/ads/video" ] && [ -d "data/games/video" ]; then
            log "Data download completed successfully!"
            
            # Count downloaded files
            AD_VIDEOS=$(find data/ads/video -name "*.mp4" | wc -l)
            GAME_VIDEOS=$(find data/games/video -name "*.mp4" | wc -l)
            log "Downloaded: $AD_VIDEOS ad videos, $GAME_VIDEOS game videos"
        else
            log "ERROR: Data download failed or incomplete"
            exit 1
        fi
    else
        log "S3 download skipped (not in AWS mode or AWS CLI not available)"
    fi
}

# Function to validate data
validate_data() {
    log "Validating data structure..."
    
    # Check if data directories exist
    if [ ! -d "data/ads/video" ] || [ ! -d "data/games/video" ]; then
        log "ERROR: Data directories not found!"
        log "Expected structure:"
        log "  data/ads/video/     (MP4 files)"
        log "  data/ads/audio/     (WAV files)"
        log "  data/games/video/   (MP4 files)"
        log "  data/games/audio/   (WAV files)"
        log ""
        log "If running on AWS, make sure to:"
        log "  1. Upload data to S3 first: aws s3 sync data/ s3://your-bucket/data/"
        log "  2. Set correct bucket name: ./run_training.sh aws your-bucket-name"
        exit 1
    fi
    
    # Count files
    AD_VIDEOS=$(find data/ads/video -name "*.mp4" | wc -l)
    GAME_VIDEOS=$(find data/games/video -name "*.mp4" | wc -l)
    AD_AUDIOS=$(find data/ads/audio -name "*.wav" | wc -l)
    GAME_AUDIOS=$(find data/games/audio -name "*.wav" | wc -l)
    
    log "Found $AD_VIDEOS ad videos and $AD_AUDIOS ad audios"
    log "Found $GAME_VIDEOS game videos and $GAME_AUDIOS game audios"
    
    if [ $AD_VIDEOS -eq 0 ] || [ $GAME_VIDEOS -eq 0 ]; then
        log "ERROR: No video files found!"
        exit 1
    fi
}

# Function to start GPU monitoring
start_monitoring() {
    if command -v nvidia-smi &> /dev/null; then
        log "Starting GPU monitoring..."
        nvidia-smi -l 1 > "gpu_monitor_${TIMESTAMP}.log" &
        GPU_MONITOR_PID=$!
    fi
}

# Function to stop monitoring
stop_monitoring() {
    if [ ! -z "$GPU_MONITOR_PID" ]; then
        log "Stopping GPU monitoring..."
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
}

# Function to save results
save_results() {
    log "Saving results..."
    
    # Create results directory
    RESULTS_DIR="results_${TIMESTAMP}"
    mkdir -p "$RESULTS_DIR"
    
    # Copy model files
    if [ -f "cost_optimized_best.pth" ]; then
        cp cost_optimized_best.pth "$RESULTS_DIR/"
        log "Saved best model: cost_optimized_best.pth"
    fi
    
    if [ -f "cost_optimized_final.pth" ]; then
        cp cost_optimized_final.pth "$RESULTS_DIR/"
        log "Saved final model: cost_optimized_final.pth"
    fi
    
    # Copy logs
    cp "$LOG_FILE" "$RESULTS_DIR/"
    if [ -f "gpu_monitor_${TIMESTAMP}.log" ]; then
        cp "gpu_monitor_${TIMESTAMP}.log" "$RESULTS_DIR/"
    fi
    
    # Copy any plots or visualizations
    if ls *.png 1> /dev/null 2>&1; then
        cp *.png "$RESULTS_DIR/"
        log "Saved plots"
    fi
    
    log "Results saved to: $RESULTS_DIR"
}

# Function to upload to S3 (AWS only)
upload_to_s3() {
    if [ "$1" = "aws" ] && command -v aws &> /dev/null; then
        log "Uploading results to S3..."
        
        if [ "$S3_BUCKET_NAME" != "your-ad-game-data-bucket" ]; then
            aws s3 cp "$RESULTS_DIR" "s3://$S3_BUCKET_NAME/results/$TIMESTAMP/" --recursive
            log "Results uploaded to s3://$S3_BUCKET_NAME/results/$TIMESTAMP/"
        else
            log "WARNING: S3 bucket name not configured. Skipping upload."
            log "To enable upload, run: ./run_training.sh aws your-bucket-name"
        fi
    fi
}

# Function to create auto-stop script (AWS only)
setup_auto_stop() {
    if [ "$1" = "aws" ]; then
        log "Setting up auto-stop for AWS (20 hours)..."
        cat > auto_stop.sh << 'EOF'
#!/bin/bash
# Auto-stop script for AWS cost control
sleep 72000  # 20 hours
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 stop-instances --instance-ids $INSTANCE_ID
echo "Auto-stopping instance $INSTANCE_ID"
EOF
        chmod +x auto_stop.sh
        nohup ./auto_stop.sh > auto_stop.log 2>&1 &
        log "Auto-stop scheduled"
    fi
}

# Main execution
main() {
    local mode=${1:-local}
    
    log "========================================"
    log "Starting Ad/Game Classifier Training"
    log "Mode: $mode"
    log "S3 Bucket: $S3_BUCKET_NAME"
    log "Timestamp: $TIMESTAMP"
    log "========================================"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # System checks
    check_gpu
    check_resources
    
    # Install dependencies if needed
    if [ "$mode" = "aws" ]; then
        install_dependencies
    fi
    
    # Download data from S3 if needed
    download_data_from_s3 "$mode"
    
    # Validate data
    validate_data
    
    # Setup auto-stop for AWS
    setup_auto_stop "$mode"
    
    # Start monitoring
    start_monitoring
    
    # Run training
    log "Starting training..."
    python cost_optimized_train.py 2>&1 | tee -a "$LOG_FILE"
    
    # Stop monitoring
    stop_monitoring
    
    # Save results
    save_results
    
    # Upload to S3 if on AWS
    upload_to_s3 "$mode"
    
    log "========================================"
    log "Training completed successfully!"
    log "Check results in: $RESULTS_DIR"
    log "Check logs in: $LOG_FILE"
    log "========================================"
}

# Handle script arguments
case "${1:-local}" in
    "local"|"aws")
        main "$1"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [local|aws] [s3-bucket-name]"
        echo ""
        echo "Options:"
        echo "  local              Run training locally (default)"
        echo "  aws                Run training on AWS with additional setup"
        echo "  s3-bucket-name     S3 bucket name for data download/upload"
        echo "  help               Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                           # Run locally"
        echo "  $0 local                     # Run locally"
        echo "  $0 aws                       # Run on AWS (uses default bucket)"
        echo "  $0 aws my-data-bucket        # Run on AWS with specific bucket"
        echo ""
        echo "S3 Data Setup:"
        echo "  1. Upload data: aws s3 sync data/ s3://your-bucket/data/"
        echo "  2. Run training: $0 aws your-bucket"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
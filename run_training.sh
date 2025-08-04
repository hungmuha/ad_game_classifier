#!/bin/bash

# ========================================
# Ad/Game Classifier Training Script
# ========================================
# This script is designed to run both locally and on AWS
# Usage: ./run_training.sh [local|aws|s3-streaming] [s3-bucket-name]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create organized logging directory structure
LOGGING_DIR="logging"
mkdir -p "$LOGGING_DIR"

# Create timestamped session directory
SESSION_DIR="$LOGGING_DIR/session_${TIMESTAMP}"
mkdir -p "$SESSION_DIR"

# Create subdirectories for different types of logs
mkdir -p "$SESSION_DIR/training"
mkdir -p "$SESSION_DIR/monitoring"
mkdir -p "$SESSION_DIR/system"
mkdir -p "$SESSION_DIR/results"

# Log file paths
LOG_FILE="$SESSION_DIR/training/training_${TIMESTAMP}.log"
GPU_LOG_FILE="$SESSION_DIR/monitoring/gpu_monitor_${TIMESTAMP}.log"
SYSTEM_LOG_FILE="$SESSION_DIR/system/system_${TIMESTAMP}.log"

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

# Function to log system info
log_system() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SYSTEM_LOG_FILE"
}

# Function to check GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_system "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | tee -a "$SYSTEM_LOG_FILE"
    else
        log_system "No GPU detected, will use CPU"
    fi
}

# Function to check system resources
check_resources() {
    log_system "System resources:"
    log_system "CPU cores: $(nproc)"
    log_system "Memory: $(free -h | grep Mem | awk '{print $2}')"
    log_system "Disk space: $(df -h . | tail -1 | awk '{print $4}') available"
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
        pip install opencv-python librosa numpy tqdm psutil moviepy scikit-learn matplotlib boto3
    fi
    
    # Install system dependencies if on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log "Installing system dependencies..."
        sudo apt update -qq
        sudo apt install -y ffmpeg libsndfile1
    fi
}

# Function to download data from S3 (for local/aws modes)
download_data_from_s3() {
    if [ "$1" = "aws" ] && command -v aws &> /dev/null; then
        log "Checking if data needs to be downloaded from S3..."
        
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
        
        # Create local data directory if it doesn't exist
        mkdir -p "data"
        
        # Function to check if a file exists locally and has same size as S3
        check_file_exists() {
            local s3_path="$1"
            local local_path="$2"
            
            # Check if local file exists
            if [ ! -f "$local_path" ]; then
                return 1  # File doesn't exist locally
            fi
            
            # Get S3 file size
            local s3_size=$(aws s3 ls "$s3_path" 2>/dev/null | awk '{print $3}')
            if [ -z "$s3_size" ]; then
                return 1  # Can't get S3 size
            fi
            
            # Get local file size
            local local_size=$(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path" 2>/dev/null)
            if [ -z "$local_size" ]; then
                return 1  # Can't get local size
            fi
            
            # Compare sizes
            if [ "$s3_size" = "$local_size" ]; then
                return 0  # Files match
            else
                return 1  # Sizes don't match
            fi
        }
        
        # Download with intelligent sync
        log "Starting intelligent S3 sync (will skip existing files)..."
        log "S3 Bucket: $S3_BUCKET_NAME"
        log "S3 Path: s3://$S3_BUCKET_NAME/$S3_DATA_PATH/"
        
        # Use aws s3 sync with --size-only flag for better efficiency
        # This will only download files that don't exist or have different sizes
        aws s3 sync "s3://$S3_BUCKET_NAME/$S3_DATA_PATH/" "data/" --size-only
        
        # Verify download
        if [ -d "data/ads/video" ] && [ -d "data/games/video" ]; then
            log "Data sync completed successfully!"
            
            # Count files and show detailed statistics
            AD_VIDEOS=$(find data/ads/video -name "*.mp4" 2>/dev/null | wc -l)
            GAME_VIDEOS=$(find data/games/video -name "*.mp4" 2>/dev/null | wc -l)
            AD_AUDIOS=$(find data/ads/audio -name "*.wav" 2>/dev/null | wc -l)
            GAME_AUDIOS=$(find data/games/audio -name "*.wav" 2>/dev/null | wc -l)
            
            log "Local files found:"
            log "  - $AD_VIDEOS ad videos"
            log "  - $GAME_VIDEOS game videos"
            log "  - $AD_AUDIOS ad audios"
            log "  - $GAME_AUDIOS game audios"
            
            # Calculate total size of downloaded data
            if command -v du &> /dev/null; then
                TOTAL_SIZE=$(du -sh data/ 2>/dev/null | cut -f1)
                log "Total data size: $TOTAL_SIZE"
            fi
            
            # Show cost savings information
            log "Cost optimization: Only new/changed files were downloaded"
            log "This saves significant S3 transfer costs on subsequent runs"
            
        else
            log "ERROR: Data sync failed or incomplete"
            exit 1
        fi
    else
        log "S3 download skipped (not in AWS mode or AWS CLI not available)"
    fi
}

# Function to validate S3 access (for s3-streaming mode)
validate_s3_access() {
    if [ "$1" = "s3-streaming" ]; then
        log "Validating S3 access for streaming mode..."
        
        # Check if AWS CLI is available
        if ! command -v aws &> /dev/null; then
            log "ERROR: AWS CLI not found. Please install it for S3 streaming mode."
            exit 1
        fi
        
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
        
        # Count files in S3
        log "Counting files in S3..."
        AD_VIDEOS=$(aws s3 ls "s3://$S3_BUCKET_NAME/$S3_DATA_PATH/ads/video/" --recursive | grep "\.mp4$" | wc -l)
        GAME_VIDEOS=$(aws s3 ls "s3://$S3_BUCKET_NAME/$S3_DATA_PATH/games/video/" --recursive | grep "\.mp4$" | wc -l)
        
        log "Found $AD_VIDEOS ad videos and $GAME_VIDEOS game videos in S3"
        
        if [ $AD_VIDEOS -eq 0 ] || [ $GAME_VIDEOS -eq 0 ]; then
            log "ERROR: No video files found in S3!"
            exit 1
        fi
        
        log "S3 streaming mode validated successfully!"
    fi
}

# Function to validate data (for local/aws modes)
validate_data() {
    if [ "$1" = "s3-streaming" ]; then
        return 0  # Skip local validation for S3 streaming
    fi
    
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
        nvidia-smi -l 1 > "$GPU_LOG_FILE" &
        GPU_MONITOR_PID=$!
        log "GPU monitoring PID: $GPU_MONITOR_PID"
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
    
    # Create results directory within session
    RESULTS_DIR="$SESSION_DIR/results"
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
    
    if [ -f "s3_streaming_best.pth" ]; then
        cp s3_streaming_best.pth "$RESULTS_DIR/"
        log "Saved S3 streaming best model: s3_streaming_best.pth"
    fi
    
    if [ -f "s3_streaming_final.pth" ]; then
        cp s3_streaming_final.pth "$RESULTS_DIR/"
        log "Saved S3 streaming final model: s3_streaming_final.pth"
    fi
    
    # Copy any plots or visualizations
    if ls *.png 1> /dev/null 2>&1; then
        cp *.png "$RESULTS_DIR/"
        log "Saved plots"
    fi
    
    # Create session summary
    cat > "$SESSION_DIR/session_summary.txt" << EOF
Training Session Summary
=======================
Session ID: ${TIMESTAMP}
Start Time: $(date)
Mode: ${1:-local}
S3 Bucket: $S3_BUCKET_NAME

Files Created:
- Training Log: $LOG_FILE
- GPU Monitor: $GPU_LOG_FILE
- System Log: $SYSTEM_LOG_FILE
- Results: $RESULTS_DIR/

Directory Structure:
$SESSION_DIR/
├── training/
│   └── training_${TIMESTAMP}.log
├── monitoring/
│   └── gpu_monitor_${TIMESTAMP}.log
├── system/
│   └── system_${TIMESTAMP}.log
└── results/
    └── [model files and plots]

EOF

    log "Results saved to: $RESULTS_DIR"
    log "Session summary: $SESSION_DIR/session_summary.txt"
    log "All logs available in: $SESSION_DIR/"
}

# Function to upload to S3 (AWS only)
upload_to_s3() {
    if [ "$1" = "aws" ] || [ "$1" = "s3-streaming" ]; then
        if command -v aws &> /dev/null; then
            log "Uploading results to S3..."
            
            if [ "$S3_BUCKET_NAME" != "your-ad-game-data-bucket" ]; then
                aws s3 cp "$SESSION_DIR" "s3://$S3_BUCKET_NAME/results/$TIMESTAMP/" --recursive
                log "Results uploaded to s3://$S3_BUCKET_NAME/results/$TIMESTAMP/"
            else
                log "WARNING: S3 bucket name not configured. Skipping upload."
                log "To enable upload, run: ./run_training.sh $1 your-bucket-name"
            fi
        fi
    fi
}

# Function to create auto-stop script (AWS only) - DISABLED
setup_auto_stop() {
    if [ "$1" = "aws" ] || [ "$1" = "s3-streaming" ]; then
        log "Auto-stop DISABLED for long training sessions..."
        log "Training will run until completion or manual stop"
        
        # Comment out the auto-stop setup for long training
        # log "Setting up auto-stop for AWS (20 hours)..."
        # cat > auto_stop.sh << 'EOF'
        # #!/bin/bash
        # # Auto-stop script for AWS cost control
        # sleep 72000  # 20 hours
        # INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
        # aws ec2 stop-instances --instance-ids $INSTANCE_ID
        # echo "Auto-stopping instance $INSTANCE_ID"
        # EOF
        # chmod +x auto_stop.sh
        # nohup ./auto_stop.sh > auto_stop.log 2>&1 &
        # log "Auto-stop scheduled"
    fi
}

# Function to find the latest checkpoint
find_latest_checkpoint() {
    # Look for all checkpoint types and find the most recent one
    local safety_checkpoints=$(find . -name "cost_optimized_safety_checkpoint_*.pth" 2>/dev/null)
    local regular_checkpoints=$(find . -name "cost_optimized_checkpoint_*.pth" 2>/dev/null)
    local best_models=$(find . -name "cost_optimized_best_epoch_*.pth" 2>/dev/null)
    
    local latest_checkpoint=""
    local latest_epoch=0
    
    # Function to extract epoch from filename
    extract_epoch() {
        local filename="$1"
        if [[ "$filename" =~ epoch_([0-9]+) ]]; then
            echo "${BASH_REMATCH[1]}"
        else
            echo "0"
        fi
    }
    
    # Check safety checkpoints first (most recent progress)
    if [ -n "$safety_checkpoints" ]; then
        for checkpoint in $safety_checkpoints; do
            local epoch=$(extract_epoch "$checkpoint")
            if [ "$epoch" -gt "$latest_epoch" ]; then
                latest_epoch=$epoch
                latest_checkpoint="$checkpoint"
            fi
        done
    fi
    
    # Check regular checkpoints (full training state)
    if [ -n "$regular_checkpoints" ]; then
        for checkpoint in $regular_checkpoints; do
            local epoch=$(extract_epoch "$checkpoint")
            if [ "$epoch" -gt "$latest_epoch" ]; then
                latest_epoch=$epoch
                latest_checkpoint="$checkpoint"
            fi
        done
    fi
    
    # Only use best models if no checkpoints found
    if [ -z "$latest_checkpoint" ] && [ -n "$best_models" ]; then
        for model in $best_models; do
            local epoch=$(extract_epoch "$model")
            if [ "$epoch" -gt "$latest_epoch" ]; then
                latest_epoch=$epoch
                latest_checkpoint="$model"
            fi
        done
    fi
    
    echo "$latest_checkpoint"
}

# Function to run training based on mode
run_training() {
    local mode=$1
    local specific_checkpoint=$2
    
    if [ "$mode" = "s3-streaming" ]; then
        log "Running S3 streaming training..."
        python s3_streaming_train.py 2>&1 | tee -a "$LOG_FILE"
    else
        log "Running validation-enhanced training..."
        
        # Determine which checkpoint to use
        local checkpoint_to_use=""
        
        if [ -n "$specific_checkpoint" ]; then
            # Use the specific checkpoint provided
            if [ -f "$specific_checkpoint" ]; then
                checkpoint_to_use="$specific_checkpoint"
                log "Using specified checkpoint: $checkpoint_to_use"
            else
                log "ERROR: Specified checkpoint '$specific_checkpoint' not found!"
                exit 1
            fi
        else
            # Find the latest checkpoint automatically
            checkpoint_to_use=$(find_latest_checkpoint)
            
            if [ -n "$checkpoint_to_use" ]; then
                log "Found latest checkpoint: $checkpoint_to_use"
                log "Resuming training with validation from checkpoint..."
            else
                log "No checkpoint found. Starting fresh training with validation..."
            fi
        fi
        
        # Run training with or without checkpoint
        if [ -n "$checkpoint_to_use" ]; then
            python cost_optimized_train.py --resume "$checkpoint_to_use" --upload-s3 --s3-bucket "$S3_BUCKET_NAME" 2>&1 | tee -a "$LOG_FILE"
        else
            python cost_optimized_train.py --upload-s3 --s3-bucket "$S3_BUCKET_NAME" 2>&1 | tee -a "$LOG_FILE"
        fi
    fi
}

# Main execution
main() {
    local mode=${1:-local}
    local specific_checkpoint=$2 # Pass the specific checkpoint file if provided
    
    log "========================================"
    log "Starting Ad/Game Classifier Training"
    log "Mode: $mode"
    log "S3 Bucket: $S3_BUCKET_NAME"
    log "Timestamp: $TIMESTAMP"
    log "Session Directory: $SESSION_DIR"
    log "========================================"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # System checks
    check_gpu
    check_resources
    
    # Install dependencies if needed
    if [ "$mode" = "aws" ] || [ "$mode" = "s3-streaming" ]; then
        install_dependencies
    fi
    
    # Handle data based on mode
    if [ "$mode" = "s3-streaming" ]; then
        validate_s3_access "$mode"
    else
        download_data_from_s3 "$mode"
        validate_data "$mode"
    fi
    
    # DISABLED: Auto-stop for long training sessions
    # setup_auto_stop "$mode"
    
    # Start monitoring
    start_monitoring
    
    # Run training
    log "Starting training..."
    run_training "$mode" "$specific_checkpoint"
    
    # Stop monitoring
    stop_monitoring
    
    # Save results
    save_results
    
    # Upload to S3 if on AWS
    upload_to_s3 "$mode"
    
    log "========================================"
    log "Training completed successfully!"
    log "Check results in: $SESSION_DIR/results/"
    log "Check logs in: $SESSION_DIR/"
    log "Session summary: $SESSION_DIR/session_summary.txt"
    log "========================================"
}

# Handle script arguments
case "${1:-local}" in
    "local"|"aws"|"s3-streaming")
        # Check if a specific checkpoint was provided as the last argument
        local mode="$1"
        local s3_bucket="${2:-your-ad-game-data-bucket}"
        local specific_checkpoint=""
        
        # Check if the last argument looks like a checkpoint file
        if [ $# -gt 2 ] && [[ "${!#}" == *.pth ]]; then
            specific_checkpoint="${!#}"
            # Remove the checkpoint from arguments so it doesn't interfere with S3 bucket
            set -- "${@:1:$(($#-1))}"
        fi
        
        main "$mode" "$specific_checkpoint"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [local|aws|s3-streaming] [s3-bucket-name] [checkpoint-file.pth]"
        echo ""
        echo "Options:"
        echo "  local              Run training locally with validation (downloads data to disk)"
        echo "  aws                Run training on AWS with validation (downloads data to EC2)"
        echo "  s3-streaming       Run training with S3 streaming (NO data download!)"
        echo "  s3-bucket-name     S3 bucket name for data access"
        echo "  checkpoint-file.pth Specific checkpoint file to resume from"
        echo "  help               Show this help message"
        echo ""
        echo "Training Features:"
        echo "  ✅ Automatic latest checkpoint detection and resume"
        echo "  ✅ Manual checkpoint specification"
        echo "  ✅ Validation data support (separate validation dataset)"
        echo "  ✅ Overfitting prevention with validation-based early stopping"
        echo "  ✅ S3 upload of results and models"
        echo "  ✅ Comprehensive logging and monitoring"
        echo ""
        echo "Examples:"
        echo "  $0                           # Run locally, auto-detect latest checkpoint"
        echo "  $0 local                     # Run locally, auto-detect latest checkpoint"
        echo "  $0 aws                       # Run on AWS, auto-detect latest checkpoint"
        echo "  $0 aws my-data-bucket        # Run on AWS with specific bucket"
        echo "  $0 aws my-bucket cost_optimized_checkpoint_epoch_4.pth  # Use specific checkpoint"
        echo "  $0 s3-streaming              # Run with S3 streaming"
        echo "  $0 s3-streaming my-bucket    # Run S3 streaming with specific bucket"
        echo ""
        echo "Checkpoint Priority:"
        echo "  1. Safety checkpoints (most recent progress)"
        echo "  2. Regular checkpoints (full training state)"
        echo "  3. Best model files (model only, starts from epoch 1)"
        echo ""
        echo "Data Setup:"
        echo "  1. Upload training data: aws s3 sync data/ s3://your-bucket/data/"
        echo "  2. Upload validation data: aws s3 sync data/validation/ s3://your-bucket/data/validation/"
        echo "  3. Run training: $0 aws your-bucket"
        echo ""
        echo "Cost Comparison:"
        echo "  local:           Downloads data to local disk"
        echo "  aws:             Downloads 2TB to EC2 (~$180-200 transfer cost)"
        echo "  s3-streaming:    Streams from S3 (~$0-5 transfer cost)"
        echo ""
        echo "Logging Structure:"
        echo "  logging/"
        echo "  └── session_TIMESTAMP/"
        echo "      ├── training/     # Training logs"
        echo "      ├── monitoring/   # GPU/system monitoring"
        echo "      ├── system/       # System info logs"
        echo "      └── results/      # Model files and plots"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
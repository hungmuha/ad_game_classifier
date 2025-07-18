#!/bin/bash

# ========================================
# Training Monitor Script
# ========================================
# Monitors training progress and system resources

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGGING_DIR="logging"
SESSION_DIR="$LOGGING_DIR/session_${TIMESTAMP}"

# Create monitoring directory if it doesn't exist
mkdir -p "$SESSION_DIR/monitoring"

# Monitoring log file
MONITOR_LOG="$SESSION_DIR/monitoring/training_monitor_${TIMESTAMP}.log"

# Function to log monitoring info
log_monitor() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

# Function to check if training is running
check_training_process() {
    local training_pid=$(ps aux | grep -E "(python.*train|s3_streaming_train|cost_optimized_train)" | grep -v grep | awk '{print $2}' | head -1)
    if [ ! -z "$training_pid" ]; then
        echo "$training_pid"
    else
        echo ""
    fi
}

# Function to check GPU usage
check_gpu_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    else
        echo "N/A,N/A,N/A,N/A"
    fi
}

# Function to check system resources
check_system_resources() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local disk_usage=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    
    echo "$cpu_usage,$memory_usage,$disk_usage"
}

# Function to check training progress
check_training_progress() {
    local latest_log=$(find "$LOGGING_DIR" -name "training_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ ! -z "$latest_log" ] && [ -f "$latest_log" ]; then
        local last_line=$(tail -1 "$latest_log" 2>/dev/null)
        if [ ! -z "$last_line" ]; then
            echo "$last_line"
        else
            echo "No training output yet"
        fi
    else
        echo "No training log found"
    fi
}

# Main monitoring loop
log_monitor "Starting training monitor..."
log_monitor "Monitoring directory: $SESSION_DIR/monitoring/"

while true; do
    # Get current timestamp
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check if training is running
    local training_pid=$(check_training_process)
    
    if [ ! -z "$training_pid" ]; then
        # Training is running
        log_monitor "Training is RUNNING (PID: $training_pid)"
        
        # Check GPU usage
        local gpu_info=$(check_gpu_usage)
        log_monitor "GPU Usage: $gpu_info"
        
        # Check system resources
        local system_info=$(check_system_resources)
        log_monitor "System Resources (CPU%, Memory%, Disk%): $system_info"
        
        # Check training progress
        local progress=$(check_training_progress)
        log_monitor "Latest Training Output: $progress"
        
    else
        # Training is not running
        log_monitor "Training is NOT RUNNING"
        
        # Check if training completed
        local latest_log=$(find "$LOGGING_DIR" -name "training_*.log" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [ ! -z "$latest_log" ] && [ -f "$latest_log" ]; then
            local last_lines=$(tail -5 "$latest_log" 2>/dev/null)
            log_monitor "Last training output:"
            echo "$last_lines" | while read line; do
                log_monitor "  $line"
            done
        fi
    fi
    
    log_monitor "----------------------------------------"
    
    # Wait before next check
    sleep 30
done
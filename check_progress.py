#!/usr/bin/env python3
"""
Training Progress Checker
=========================
Monitors training progress and provides status updates
"""

import os
import json
import time
import glob
import subprocess
import psutil
from datetime import datetime
from pathlib import Path

class TrainingProgressChecker:
    def __init__(self):
        self.logging_dir = "logging"
        self.session_dir = None
        self.monitor_log = None
        
    def find_latest_session(self):
        """Find the latest training session directory"""
        if not os.path.exists(self.logging_dir):
            return None
            
        sessions = glob.glob(os.path.join(self.logging_dir, "session_*"))
        if not sessions:
            return None
            
        # Sort by creation time and get the latest
        latest_session = max(sessions, key=os.path.getctime)
        return latest_session
    
    def setup_monitoring(self):
        """Setup monitoring for current session"""
        self.session_dir = self.find_latest_session()
        if self.session_dir:
            self.monitor_log = os.path.join(self.session_dir, "monitoring", "progress_checker.log")
            os.makedirs(os.path.dirname(self.monitor_log), exist_ok=True)
            print(f"Monitoring session: {self.session_dir}")
        else:
            print("No training session found")
    
    def log_status(self, message):
        """Log status message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        if self.monitor_log:
            with open(self.monitor_log, 'a') as f:
                f.write(log_message + '\n')
    
    def check_training_process(self):
        """Check if training process is running"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'])
                if any(x in cmdline for x in ['s3_streaming_train.py', 'cost_optimized_train.py', 'run_training.sh']):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def check_gpu_usage(self):
        """Check GPU usage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "N/A,N/A,N/A,N/A"
        except FileNotFoundError:
            return "N/A,N/A,N/A,N/A"
    
    def check_system_resources(self):
        """Check system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def get_latest_training_log(self):
        """Get the latest training log content"""
        if not self.session_dir:
            return None
            
        training_logs = glob.glob(os.path.join(self.session_dir, "training", "training_*.log"))
        if not training_logs:
            return None
            
        latest_log = max(training_logs, key=os.path.getctime)
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                return {
                    'file': latest_log,
                    'lines': lines[-10:],  # Last 10 lines
                    'total_lines': len(lines)
                }
        except FileNotFoundError:
            return None
    
    def get_gpu_monitor_log(self):
        """Get GPU monitoring log"""
        if not self.session_dir:
            return None
            
        gpu_logs = glob.glob(os.path.join(self.session_dir, "monitoring", "gpu_monitor_*.log"))
        if not gpu_logs:
            return None
            
        latest_gpu_log = max(gpu_logs, key=os.path.getctime)
        
        try:
            with open(latest_gpu_log, 'r') as f:
                lines = f.readlines()
                return {
                    'file': latest_gpu_log,
                    'lines': lines[-5:],  # Last 5 lines
                    'total_lines': len(lines)
                }
        except FileNotFoundError:
            return None
    
    def display_status(self):
        """Display current training status"""
        print("\n" + "="*60)
        print("TRAINING PROGRESS CHECKER")
        print("="*60)
        
        # Check training process
        training_pid = self.check_training_process()
        if training_pid:
            self.log_status(f"Training is RUNNING (PID: {training_pid})")
        else:
            self.log_status("Training is NOT RUNNING")
        
        # Check GPU usage
        gpu_info = self.check_gpu_usage()
        self.log_status(f"GPU Usage: {gpu_info}")
        
        # Check system resources
        system_info = self.check_system_resources()
        self.log_status(f"System Resources:")
        self.log_status(f"  CPU: {system_info['cpu_percent']:.1f}%")
        self.log_status(f"  Memory: {system_info['memory_percent']:.1f}% ({system_info['memory_used_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB)")
        self.log_status(f"  Disk: {system_info['disk_percent']:.1f}% (Free: {system_info['disk_free_gb']:.1f}GB)")
        
        # Check training logs
        training_log = self.get_latest_training_log()
        if training_log:
            self.log_status(f"Training Log: {training_log['file']}")
            self.log_status(f"Total log lines: {training_log['total_lines']}")
            self.log_status("Latest training output:")
            for line in training_log['lines']:
                self.log_status(f"  {line.rstrip()}")
        else:
            self.log_status("No training log found")
        
        # Check GPU monitoring
        gpu_log = self.get_gpu_monitor_log()
        if gpu_log:
            self.log_status(f"GPU Monitor Log: {gpu_log['file']}")
            self.log_status(f"GPU monitor entries: {gpu_log['total_lines']}")
        
        print("-"*60)
    
    def run_continuous_monitoring(self, interval=30):
        """Run continuous monitoring"""
        self.log_status("Starting continuous monitoring...")
        self.log_status(f"Check interval: {interval} seconds")
        
        try:
            while True:
                self.display_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.log_status("Monitoring stopped by user")
    
    def run_single_check(self):
        """Run a single status check"""
        self.display_status()

def main():
    checker = TrainingProgressChecker()
    checker.setup_monitoring()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        checker.run_continuous_monitoring()
    else:
        checker.run_single_check()

if __name__ == "__main__":
    main()
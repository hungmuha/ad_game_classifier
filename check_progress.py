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
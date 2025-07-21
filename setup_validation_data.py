#!/usr/bin/env python3
"""
Setup Validation Data
====================
Helps set up validation data structure and upload to S3.
"""

import os
import argparse
import subprocess
import boto3
from datetime import datetime

def check_validation_structure():
    """Check if validation data structure exists"""
    val_dirs = [
        'data/validation/ads/video',
        'data/validation/ads/audio', 
        'data/validation/games/video',
        'data/validation/games/audio'
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for dir_path in val_dirs:
        if os.path.exists(dir_path):
            existing_dirs.append(dir_path)
        else:
            missing_dirs.append(dir_path)
    
    return existing_dirs, missing_dirs

def create_validation_structure():
    """Create validation data directory structure"""
    val_dirs = [
        'data/validation/ads/video',
        'data/validation/ads/audio', 
        'data/validation/games/video',
        'data/validation/games/audio'
    ]
    
    for dir_path in val_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def count_files_in_dirs(dirs):
    """Count files in directories"""
    counts = {}
    for dir_path in dirs:
        if os.path.exists(dir_path):
            if 'video' in dir_path:
                files = len([f for f in os.listdir(dir_path) if f.endswith('.mp4')])
            else:
                files = len([f for f in os.listdir(dir_path) if f.endswith('.wav')])
            counts[dir_path] = files
    return counts

def upload_validation_to_s3(bucket_name, local_path="data/validation"):
    """Upload validation data to S3"""
    try:
        s3_client = boto3.client('s3')
        
        # Upload validation data
        subprocess.run([
            'aws', 's3', 'sync', 
            local_path, 
            f's3://{bucket_name}/data/validation/',
            '--progress'
        ], check=True)
        
        print(f"✅ Uploaded validation data to s3://{bucket_name}/data/validation/")
        return True
    except Exception as e:
        print(f"❌ Failed to upload validation data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup validation data structure')
    parser.add_argument('--create', action='store_true', help='Create validation directory structure')
    parser.add_argument('--upload', type=str, help='Upload validation data to S3 bucket')
    parser.add_argument('--check', action='store_true', help='Check current validation data structure')
    
    args = parser.parse_args()
    
    print("🔍 Validation Data Setup")
    print("=" * 40)
    
    if args.check:
        print("\n📁 Checking validation data structure...")
        existing_dirs, missing_dirs = check_validation_structure()
        
        if existing_dirs:
            print("✅ Existing directories:")
            for dir_path in existing_dirs:
                counts = count_files_in_dirs([dir_path])
                print(f"   - {dir_path}: {counts.get(dir_path, 0)} files")
        
        if missing_dirs:
            print("❌ Missing directories:")
            for dir_path in missing_dirs:
                print(f"   - {dir_path}")
        
        if not missing_dirs:
            print("\n✅ Validation data structure is complete!")
        else:
            print("\n⚠️  Run with --create to create missing directories")
    
    if args.create:
        print("\n📁 Creating validation data structure...")
        create_validation_structure()
        print("\n✅ Validation directory structure created!")
        print("\n📋 Next steps:")
        print("1. Add validation video files to data/validation/ads/video/")
        print("2. Add validation audio files to data/validation/ads/audio/")
        print("3. Add validation video files to data/validation/games/video/")
        print("4. Add validation audio files to data/validation/games/audio/")
        print("5. Run with --upload BUCKET_NAME to upload to S3")
    
    if args.upload:
        print(f"\n📤 Uploading validation data to S3 bucket: {args.upload}")
        if upload_validation_to_s3(args.upload):
            print("✅ Validation data uploaded successfully!")
        else:
            print("❌ Upload failed!")
    
    if not any([args.check, args.create, args.upload]):
        print("Usage examples:")
        print("  python setup_validation_data.py --check")
        print("  python setup_validation_data.py --create")
        print("  python setup_validation_data.py --upload your-bucket-name")

if __name__ == "__main__":
    main() 
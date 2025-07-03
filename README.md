# Ad/Game Video Classifier

A machine learning system that automatically classifies video content as either advertisements or sports games using multimodal deep learning (video + audio analysis).

## Project Overview

This project demonstrates a complete ML pipeline for video classification, from data preprocessing to cloud deployment. It's designed to handle large-scale video datasets efficiently while maintaining cost-effectiveness.

### Key Features
- **Multimodal Analysis**: Combines video frames and audio features for robust classification
- **Cost-Optimized Training**: Memory-efficient architecture designed for cloud deployment
- **Production-Ready**: Complete AWS deployment pipeline with automated scripts
- **Scalable**: Handles 200+ hours of video content (~200GB dataset)

##️ Architecture

### Model Architecture
- **Video Branch**: MobileNetV2 backbone for frame analysis
- **Audio Branch**: MFCC features for audio classification
- **Fusion**: Late fusion of video and audio features
- **Output**: Binary classification (Ad vs Game)

### Technical Stack
- **Deep Learning**: PyTorch with mixed precision training
- **Computer Vision**: OpenCV for video processing
- **Audio Processing**: Librosa for MFCC extraction
- **Cloud**: AWS EC2 with GPU instances
- **Data Storage**: AWS S3 for scalable data management

## 📁 Repository Structure

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/your-username/ad_game_classifier.git
cd ad_game_classifier

# Install dependencies
pip install pipenv
pipenv install

# Make training script executable
chmod +x run_training.sh

# Run local training (cost-optimized version)
./run_training.sh local
```

### AWS Deployment
```bash
# Upload data to S3
aws s3 sync data/ s3://your-bucket/data/

# Launch EC2 instance and run training
./run_training.sh aws your-bucket-name
```

## 💡 Key Technical Decisions

### 1. Cost Optimization
- **MobileNetV2**: Lightweight CNN backbone for efficient inference
- **Reduced Frame Rate**: 4 FPS instead of 8 FPS to save memory
- **Smaller Frame Size**: 96x96 instead of 112x112
- **Mixed Precision**: 2x speedup with automatic mixed precision
- **Early Stopping**: Prevents overfitting and reduces training time

### 2. Multimodal Approach
- **Video Analysis**: Captures visual patterns and motion
- **Audio Analysis**: Identifies audio signatures (commentary vs music)
- **Late Fusion**: Combines features for robust classification

### 3. Production Considerations
- **Automated Pipeline**: Single script handles setup, training, and deployment
- **Cloud-Native**: Designed for AWS with S3 integration
- **Monitoring**: Built-in GPU and system resource tracking
- **Cost Control**: Auto-stop functionality prevents runaway costs

## ⚙️ Configuration

### Training Scripts
The project includes two training scripts with different configurations:

#### `cost_optimized_train.py` (Recommended for Cloud)
- **Batch Size**: 8 (optimized for GPU memory)
- **Epochs**: 8 with early stopping
- **Frame Rate**: 4 FPS
- **Frame Size**: 96x96
- **MFCC Features**: 32 mel coefficients
- **Learning Rate**: 2e-4
- **Mixed Precision**: Enabled

#### `train.py` (Full-featured)
- **Batch Size**: 2 (higher memory usage)
- **Epochs**: 10
- **Frame Rate**: 6 FPS
- **Frame Size**: 112x112
- **MFCC Features**: 40 mel coefficients
- **Learning Rate**: 1e-4

### Hyperparameter Tuning
To modify training parameters, edit the configuration section at the top of the respective training script:

```python
# In cost_optimized_train.py or train.py
BATCH_SIZE = 8          # Adjust based on GPU memory
EPOCHS = 8              # Training epochs
FPS = 4                 # Frames per second
FRAME_SIZE = (96, 96)   # Frame dimensions
MFCC_N_MELS = 32        # Audio features
LEARNING_RATE = 2e-4    # Learning rate
```

##  Performance & Scalability

### Model Performance
- **Input**: 10-second video clips with corresponding audio
- **Output**: Binary classification (Ad/Game) with confidence score
- **Training Time**: ~8 epochs with early stopping
- **Memory Usage**: Optimized for 16GB GPU memory

### Scalability Features
- **Batch Processing**: Configurable batch sizes for different hardware
- **Multi-GPU Support**: Ready for distributed training
- **Cloud Deployment**: Automated AWS setup and execution
- **Data Pipeline**: Efficient loading and preprocessing

## 🔧 Development Workflow

### For Developers
1. **Local Testing**: Use `./run_training.sh local` for development
2. **Data Preparation**: Organize videos in `data/ads/` and `data/games/`
3. **Model Tuning**: Modify configuration parameters in training scripts
4. **Cloud Testing**: Use `./run_training.sh aws bucket-name` for production

## 📈 Business Impact

### Use Cases
- **Content Moderation**: Automatically flag advertisements in video streams
- **Sports Analytics**: Separate game content from commercial breaks
- **Media Processing**: Large-scale video classification for content libraries
- **Ad Detection**: Identify and analyze advertising content

### Cost Benefits
- **Training Cost**: ~$15-20 for complete model training
- **Inference Cost**: Optimized for real-time processing
- **Scalability**: Handles 200+ hours of content efficiently

## 🛠️ Technical Skills Demonstrated

### Machine Learning
- Deep Learning with PyTorch
- Multimodal model architecture
- Hyperparameter optimization
- Model evaluation and validation

### Software Engineering
- Python development
- Shell scripting and automation
- Configuration management
- Error handling and logging

### Cloud & DevOps
- AWS EC2 and S3
- Infrastructure as code
- Monitoring and alerting
- Cost optimization

### Data Engineering
- Large-scale data processing
- Video and audio preprocessing
- Efficient data pipelines
- Storage optimization

## 📚 Documentation

- **[AWS Deployment Guide](aws_deployment_guide.md)**: Complete cloud deployment instructions
- **[Training Explained](training_explained.md)**: Detailed training process documentation
- **[Deployment Checklist](deployment_checklist.md)**: Pre-deployment verification steps

##  Contributing

This project is designed as a demonstration of production-ready ML systems. Key areas for improvement:
- Model architecture optimization
- Additional data augmentation techniques
- Real-time inference pipeline
- Multi-class classification support

---

**Note**: This project is designed for educational and demonstration purposes, showcasing best practices in ML system development and cloud deployment.

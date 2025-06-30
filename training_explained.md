# Machine Learning Training Explained (Beginner-Friendly)

## 🎯 What We're Doing
We're teaching a computer to tell the difference between NFL game footage and TV commercials.

## 📊 Your Data
- **200 hours** of video (100 hours ads + 100 hours games)
- Each clip is **10 seconds** long
- Format: MP4 video + WAV audio

## 🧠 How the Model Works

### 1. Video Processing
```
10-second video → Extract 40 frames → Each frame becomes 96×96 pixels
```
**Why 40 frames?** We sample 4 frames per second to capture movement without using too much memory.

### 2. Audio Processing
```
10-second audio → Extract 32 sound features (MFCC)
```
**What are MFCC features?** They represent the "fingerprint" of the sound (like how loud, what pitch, etc.)

### 3. The Neural Network
```
Video (40 frames) → MobileNetV2 → 128 numbers
Audio (32 features) → Small network → 32 numbers
Combine (160 numbers) → Final network → 1 number (0 or 1)
```

## 🎓 Training Process

### Phase 1: Learning (Epochs 1-3)
- Model makes random guesses
- Gets lots of feedback
- Loss is high (many mistakes)

### Phase 2: Improving (Epochs 4-6)
- Model starts recognizing patterns
- Fewer mistakes
- Loss decreases

### Phase 3: Fine-tuning (Epochs 7-8)
- Model gets very good
- Loss stabilizes
- Early stopping if no improvement

## 📈 What "Loss" Means
- **Loss = How many mistakes the model makes**
- **Lower loss = Better performance**
- **Goal**: Get loss as close to 0 as possible

## 🔄 Training Loop Explained

```python
for each epoch (training cycle):
    for each batch of 8 clips:
        1. Load 8 video clips + audio + labels
        2. Model predicts: ad or game?
        3. Compare predictions with true labels
        4. Calculate how wrong the model was (loss)
        5. Update model to be less wrong next time
    6. Save the best model so far
```

## 💰 Cost Breakdown

### What You're Paying For:
1. **GPU Time**: Computer doing calculations
2. **Memory**: Storing video frames and model
3. **Storage**: Saving your data and model

### Cost Optimization:
- **Smaller frames** (96×96 vs 112×112) = Less memory
- **Fewer frames** (40 vs 80) = Faster processing
- **Mixed precision** = 2x speedup
- **Early stopping** = Stop when good enough

## 🎯 Expected Results

### Training Progress:
- **Epoch 1**: Loss ~0.6 (40% accuracy)
- **Epoch 4**: Loss ~0.3 (70% accuracy)
- **Epoch 8**: Loss ~0.2 (80%+ accuracy)

### Final Model:
- Should achieve **80-90% accuracy**
- Can process new videos in **real-time**
- File size: ~50-100MB

## 🚀 What Happens After Training

1. **Save the model** as a file
2. **Test on new videos** it hasn't seen
3. **Deploy** to classify live streams
4. **Monitor performance** and retrain if needed

## 🔍 Key Terms Explained

- **Epoch**: One complete pass through all training data
- **Batch**: Group of clips processed together (8 in our case)
- **Loss**: Measure of how wrong the model is
- **Accuracy**: Percentage of correct predictions
- **GPU**: Special computer chip for fast calculations
- **Model**: The "brain" that learns to classify videos 

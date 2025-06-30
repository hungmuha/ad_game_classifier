# 🚀 AWS Deployment Checklist

## ✅ Pre-Deployment (Local)

- [ ] **Local Testing**: Run `python test_local.py` - all tests pass
- [ ] **Data Preparation**: 200 hours of MP4/WAV files ready
- [ ] **Code Ready**: `cost_optimized_train.py` finalized
- [ ] **AWS Account**: Set up with billing alerts
- [ ] **AWS CLI**: Installed and configured (`aws configure`)

## 📤 Step 1: Data Upload (2-4 hours)

- [ ] **Create S3 Bucket**: `aws s3 mb s3://your-ad-game-data-bucket`
- [ ] **Upload Data**: `aws s3 sync data/ s3://your-ad-game-data-bucket/data/ --progress`
- [ ] **Verify Upload**: Check all files transferred correctly

## 🖥️ Step 2: Launch Instance (5 minutes)

- [ ] **Run Setup Script**: `chmod +x aws_setup.sh && ./aws_setup.sh`
- [ ] **Note Instance Details**: IP address and instance ID
- [ ] **Test Connection**: `ssh -i your-key.pem ubuntu@INSTANCE_IP`

## 🔧 Step 3: Instance Setup (10 minutes)

- [ ] **Upload Code**: Copy training scripts to instance
- [ ] **Download Data**: `aws s3 sync s3://your-bucket/data/ data/`
- [ ] **Test GPU**: `nvidia-smi` shows T4 GPU
- [ ] **Test Dependencies**: `python -c "import torch; print(torch.cuda.is_available())"`

## 🎯 Step 4: Start Training (15-25 hours)

- [ ] **Start Training**: `./run_training.sh`
- [ ] **Monitor Progress**: `tail -f training.log`
- [ ] **Check GPU Usage**: `watch -n 1 nvidia-smi`
- [ ] **Monitor Costs**: Check AWS billing dashboard

## 💾 Step 5: Save Results (5 minutes)

- [ ] **Download Models**: `aws s3 sync s3://your-bucket/models/ ./models/`
- [ ] **Download Logs**: `aws s3 sync s3://your-bucket/logs/ ./logs/`
- [ ] **Terminate Instance**: `aws ec2 terminate-instances --instance-ids INSTANCE_ID`

## 📊 Expected Timeline

| Step | Duration | Cost |
|------|----------|------|
| Data Upload | 2-4 hours | $2-5 |
| Instance Setup | 10 minutes | $0.09 |
| Training | 15-25 hours | $7.89-13.15 |
| **Total** | **17-29 hours** | **$10-18** |

## 🚨 Cost Control Measures

- [ ] **Auto-stop**: Instance stops after 20 hours
- [ ] **Billing Alerts**: Set up CloudWatch alarms
- [ ] **Spot Instances**: Consider for 60-90% savings
- [ ] **Monitor Usage**: Check AWS console regularly

## 🔍 Monitoring Commands

```bash
# Check training progress
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check system resources
htop

# Monitor costs
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-02 --granularity DAILY --metrics BlendedCost
```

## 📱 Quick Start Commands

```bash
# 1. Upload data
aws s3 sync data/ s3://your-bucket/data/

# 2. Launch instance
./aws_setup.sh

# 3. Connect and start training
ssh -i your-key.pem ubuntu@INSTANCE_IP
cd training
./run_training.sh

# 4. Monitor
tail -f training.log

# 5. Download results
aws s3 sync s3://your-bucket/models/ ./models/
```

## 🆘 Troubleshooting

- **Out of Memory**: Reduce batch size in `cost_optimized_train.py`
- **Slow Training**: Check GPU utilization with `nvidia-smi`
- **Connection Issues**: Verify security group allows SSH
- **Data Issues**: Check S3 bucket permissions

## 🎯 Success Criteria

- [ ] Training completes without errors
- [ ] Loss decreases over epochs
- [ ] Final model achieves >80% accuracy
- [ ] Total cost < $20
- [ ] Model files saved to S3

**Ready to deploy! 🚀** 

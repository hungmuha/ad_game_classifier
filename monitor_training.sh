#!/bin/bash
# Monitor training progress

echo "=== Training Status ==="
echo "Latest log file:"
ls -t logs/training_*.log | head -1 | xargs tail -20

echo ""
echo "=== Progress Summary ==="
echo "Latest progress file:"
ls -t logs/training_*_progress.json | head -1 | xargs cat | python -m json.tool

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "=== System Resources ==="
echo "Memory:"
free -h
echo "Disk:"
df -h .
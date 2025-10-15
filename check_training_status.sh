#!/bin/bash
# Check status of ablation training

echo "Checking training status..."
echo ""

# Check if processes are running
if pgrep -f "pretrain.py" > /dev/null; then
    echo "✓ Training is still running"
    echo ""
    echo "Recent log output:"
    echo "===================="
    tail -20 ablation_40ep_training.log
else
    echo "✗ Training has completed or stopped"
    echo ""
    echo "Final log output:"
    echo "===================="
    tail -50 ablation_40ep_training.log
fi

echo ""
echo "Checkpoint directories:"
ls -lah checkpoints/Maze-30x30-hard-1k-ACT-torch/ | grep ablation


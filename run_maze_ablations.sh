#!/bin/bash
# Script to run TRM ablations on Maze dataset
# Based on ablations from Table 1 in the paper

set -e  # Exit on error

echo "================================================"
echo "Running TRM Ablations on Maze Dataset"
echo "================================================"
echo ""
echo "This will run 5 experiments:"
echo "  1. Baseline TRM (T=3, n=6, 2-layers, with EMA)"
echo "  2. No EMA ablation"
echo "  3. With ACT continue loss ablation"
echo "  4. T=2, n=2 ablation (less recursion)"
echo "  5. 4-layers, n=3 ablation (more layers, less recursion)"
echo ""
echo "Each experiment trains for 60k epochs (~24 hours on 4 GPUs)"
echo "================================================"
echo ""

# Make sure data exists
if [ ! -d "data/maze-30x30-hard-1k" ]; then
    echo "ERROR: Maze dataset not found at data/maze-30x30-hard-1k"
    echo "Please run: python dataset/build_maze_dataset.py"
    exit 1
fi

# Number of GPUs to use
NGPUS=${NGPUS:-4}
echo "Using $NGPUS GPUs (set NGPUS environment variable to change)"
echo ""

# Function to run training
run_experiment() {
    local config=$1
    local name=$2
    
    echo "================================================"
    echo "Starting: $name"
    echo "Config: $config"
    echo "================================================"
    
    if [ $NGPUS -gt 1 ]; then
        torchrun --nproc_per_node=$NGPUS pretrain.py --config-name=$config
    else
        python pretrain.py --config-name=$config
    fi
    
    echo ""
    echo "Completed: $name"
    echo ""
}

# Run ablations
run_experiment "cfg_pretrain_maze" "Baseline TRM"
run_experiment "cfg_pretrain_maze_no_ema" "No EMA Ablation"
run_experiment "cfg_pretrain_maze_with_act" "With ACT Continue Loss"
run_experiment "cfg_pretrain_maze_T2_n2" "T=2, n=2 Ablation"
run_experiment "cfg_pretrain_maze_4layers" "4-layers, n=3 Ablation"

echo "================================================"
echo "All ablations completed!"
echo "================================================"
echo ""
echo "Results can be found in the checkpoints/ directory"
echo "Check your W&B dashboard for comparative results"


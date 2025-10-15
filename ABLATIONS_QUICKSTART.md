# TRM Maze Ablations - Quick Start Guide

This guide helps you run ablation experiments on the TRM model for the Maze dataset.

## Overview

We've set up 5 ablation experiments based on Table 1 from the paper:

1. **Baseline TRM** - T=3, n=6, 2-layers, with EMA
2. **No EMA** - Tests training stability without Exponential Moving Average
3. **With ACT Continue** - Tests the full ACT Q-learning vs simplified halting
4. **T=2, n=2** - Tests impact of reduced recursion depth
5. **4-layers, n=3** - Tests "less is more" hypothesis (bigger vs deeper)

## Prerequisites

1. **Prepare the Maze dataset**:
   ```bash
   python dataset/build_maze_dataset.py --aug
   ```
   This downloads and processes the Maze-30x30-Hard dataset with augmentations.

2. **Check requirements**:
   - 4 GPUs recommended (adjust NGPUS in script if needed)
   - ~100GB disk space for checkpoints
   - Weights & Biases account for logging

## Running Experiments

### Option 1: Run All Ablations (Recommended)

Run all 5 experiments sequentially:

```bash
bash run_maze_ablations.sh
```

**Time estimate**: ~5 days total (each experiment takes ~24 hours on 4 GPUs)

### Option 2: Run Individual Experiments

Run a single experiment:

```bash
# Baseline
python pretrain.py --config-name=cfg_pretrain_maze

# Or with multiple GPUs
torchrun --nproc_per_node=4 pretrain.py --config-name=cfg_pretrain_maze
```

Available configs:
- `cfg_pretrain_maze` - Baseline
- `cfg_pretrain_maze_no_ema` - No EMA
- `cfg_pretrain_maze_with_act` - With ACT continue
- `cfg_pretrain_maze_T2_n2` - T=2, n=2
- `cfg_pretrain_maze_4layers` - 4-layers, n=3

### Option 3: Run Specific Ablation

To run just one ablation (e.g., the no-EMA experiment):

```bash
# Single GPU
python pretrain.py --config-name=cfg_pretrain_maze_no_ema

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 pretrain.py --config-name=cfg_pretrain_maze_no_ema
```

## Monitoring Progress

### During Training

All experiments log to Weights & Biases. Key metrics to watch:
- `train/loss` - Training loss
- `train/avg_steps` - Average halting steps (should be < 16)
- `train/lr` - Learning rate (warmup + cosine decay)

### After Training

Run the analysis script to compare results:

```bash
python analyze_maze_ablations.py
```

This will:
- Search for completed experiments
- Extract test accuracy and other metrics
- Create a comparison table
- Save results to `maze_ablation_results.csv`

## Expected Results

Based on Sudoku results from Table 1, we expect:

| Ablation | Effective Depth | Expected Δ |
|----------|-----------------|------------|
| Baseline | 42 layers | 0.0% (reference) |
| No EMA | 42 layers | -7.5% |
| With ACT | 42 layers | -1.3% |
| T=2, n=2 | 12 layers | -13.7% |
| 4-layers | 48 layers | -7.9% |

## Experiment Details

Each experiment:
- Trains for **60,000 epochs**
- Uses **batch size 768**
- Learning rate **1e-4** with warmup
- Evaluates every **10,000 epochs** (starting from epoch 50,000)
- Saves checkpoints after each evaluation

## File Structure

```
config/
├── cfg_pretrain_maze.yaml              # Baseline config
├── cfg_pretrain_maze_no_ema.yaml       # No EMA ablation
├── cfg_pretrain_maze_with_act.yaml     # With ACT continue
├── cfg_pretrain_maze_T2_n2.yaml        # T=2, n=2 ablation
├── cfg_pretrain_maze_4layers.yaml      # 4-layers ablation
└── arch/
    ├── trm_maze_baseline.yaml          # Baseline architecture
    ├── trm_maze_T2_n2.yaml             # T=2, n=2 architecture
    ├── trm_maze_4layers_n3.yaml        # 4-layers architecture
    └── trm_maze_with_act_continue.yaml # With ACT architecture

run_maze_ablations.sh        # Script to run all experiments
analyze_maze_ablations.py    # Script to analyze results
MAZE_ABLATIONS.md           # Detailed documentation
```

## Troubleshooting

### Out of Memory
- Reduce `global_batch_size` in config (e.g., 768 → 384)
- Use fewer GPUs and adjust batch size accordingly
- The 4-layers ablation uses more memory

### Dataset Not Found
```bash
python dataset/build_maze_dataset.py --aug
```

### Slow Training
- Check GPU utilization with `nvidia-smi`
- Ensure you're using `torchrun` for multi-GPU
- The "With ACT" ablation is intentionally 2× slower (extra forward pass)

### No Results in Analysis
- Wait for experiments to complete (check W&B dashboard)
- Run analysis script after training finishes

## Next Steps

After running experiments:

1. **Compare results** - Run `python analyze_maze_ablations.py`
2. **Visualize in W&B** - Compare loss curves, accuracy trends
3. **Analyze checkpoints** - Inspect saved models
4. **Extend ablations** - Try other variations (different T/n, learning rates, etc.)

## Questions?

See detailed documentation in:
- `MAZE_ABLATIONS.md` - Full ablation study documentation
- `paper.md` - Original TRM paper with theory
- `README.md` - General project information

## Citation

If you use these ablations in your research:

```
Less is More: Recursive Reasoning with Tiny Networks
Alexia Jolicoeur-Martineau
arXiv:2510.04871
```


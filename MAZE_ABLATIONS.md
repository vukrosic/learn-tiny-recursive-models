# TRM Ablation Study on Maze Dataset

This document describes the ablation experiments for the Tiny Recursive Model (TRM) on the Maze-Hard dataset, replicating key ablations from Table 1 of the paper.

## Important Note: TRM vs HRM

**TRM (Tiny Recursive Model)** is a simplified version of HRM that uses a **single network** instead of two:
- HRM uses two networks: `fL` (low-level) and `fH` (high-level) at different frequencies
- TRM uses **one network** for both latent reasoning and answer updates

The code still uses HRM variable names (`H_cycles`, `L_cycles`, `z_H`, `z_L`) but with TRM interpretation:
- `z_H` → `y` in paper (current solution/answer)
- `z_L` → `z` in paper (latent reasoning feature)
- `H_cycles` → `T` (deep recursion: how many times to improve the answer)
- `L_cycles` → `n` (latent recursion: how many times to update `z` before updating `y`)
- `H_layers: 0` → TRM doesn't use a separate H network
- `L_layers: 2` → The single 2-layer network used for both updates

## Dataset

- **Task**: Maze pathfinding on 30x30 grids
- **Training size**: 1,000 examples (with 8 dihedral augmentations = 8,000 examples)
- **Test size**: 1,000 examples
- **Challenge**: Find shortest path in hard mazes (path length > 110)

## Experiments

### 1. Baseline TRM
**Config**: `cfg_pretrain_maze.yaml` with `arch/trm_maze_baseline.yaml`

**Settings**:
- **T=3** (`H_cycles=3`): 3 deep recursion cycles to improve answer `y`
- **n=6** (`L_cycles=6`): 6 latent recursion steps to update `z` per cycle
- **Single 2-layer network** (`L_layers=2`, `H_layers=0`)
- With EMA (ema=True, rate=0.999)
- No ACT continue loss (no_ACT_continue=True)
- Self-attention (mlp_t=False)

**Effective depth**: T × (n+1) × L_layers = 3 × 7 × 2 = 42 layers

**Algorithm**: For each supervision step:
1. Run T-1 cycles without gradients: `for _ in range(n): z = net(x, y, z)` then `y = net(y, z)`
2. Run 1 cycle with gradients: `for _ in range(n): z = net(x, y, z)` then `y = net(y, z)`

**Expected**: Best performance (paper shows TRM-Att gets 85.3% on Maze-Hard)

---

### 2. No EMA Ablation
**Config**: `cfg_pretrain_maze_no_ema.yaml`

**Change**: `ema: False` (disable Exponential Moving Average)

**Hypothesis**: Performance should drop due to training instability and overfitting. Paper shows -7.5% on Sudoku (79.9% vs 87.4%).

**Purpose**: Tests the importance of EMA for training stability on small datasets.

---

### 3. With ACT Continue Loss
**Config**: `cfg_pretrain_maze_with_act.yaml` with `arch/trm_maze_with_act_continue.yaml`

**Change**: `no_ACT_continue: False` (enable Q-learning continue loss)

**Hypothesis**: Slight performance drop (-1.3% on Sudoku: 86.1% vs 87.4%) but requires 2× forward passes during training, making it slower.

**Purpose**: Tests whether the simplified halting mechanism (only halt signal) is sufficient vs. the full Q-learning ACT.

---

### 4. T=2, n=2 Ablation (Less Recursion)
**Config**: `cfg_pretrain_maze_T2_n2.yaml` with `arch/trm_maze_T2_n2.yaml`

**Changes**:
- H_cycles: 3 → 2
- L_cycles: 6 → 2

**Effective depth**: 2 × 3 × 2 = 12 layers (vs. 42 baseline)

**Hypothesis**: Significant performance drop due to much less recursion. Paper shows -13.7% on Sudoku (73.7% vs 87.4%).

**Purpose**: Tests the importance of deep recursion for hard reasoning tasks.

---

### 5. 4-Layers, n=3 Ablation
**Config**: `cfg_pretrain_maze_4layers.yaml` with `arch/trm_maze_4layers_n3.yaml`

**Changes**:
- L_layers: 2 → 4
- L_cycles: 6 → 3 (compensate for more layers)

**Effective depth**: 3 × 4 × 4 = 48 layers (similar to baseline's 42)

**Hypothesis**: Performance drop due to overfitting with larger network. Paper shows -7.9% on Sudoku (79.5% vs 87.4%).

**Purpose**: Tests "less is more" hypothesis - smaller networks with more recursion generalize better than larger networks.

---

## Training Details

All experiments use:
- **Batch size**: 768
- **Epochs**: 60,000
- **Learning rate**: 1e-4
- **Weight decay**: 1.0
- **Warmup steps**: 2,000
- **Eval interval**: Every 10,000 epochs
- **Min eval interval**: 50,000 epochs (start evaluation)

## Running the Experiments

### Option 1: Run all ablations sequentially
```bash
bash run_maze_ablations.sh
```

### Option 2: Run individual experiments

**Baseline**:
```bash
python pretrain.py --config-name=cfg_pretrain_maze
```

**No EMA**:
```bash
python pretrain.py --config-name=cfg_pretrain_maze_no_ema
```

**With ACT Continue**:
```bash
python pretrain.py --config-name=cfg_pretrain_maze_with_act
```

**T=2, n=2**:
```bash
python pretrain.py --config-name=cfg_pretrain_maze_T2_n2
```

**4-layers, n=3**:
```bash
python pretrain.py --config-name=cfg_pretrain_maze_4layers
```

### Multi-GPU Training
```bash
# Use torchrun for distributed training
torchrun --nproc_per_node=4 pretrain.py --config-name=cfg_pretrain_maze
```

## Expected Results

Based on the Sudoku results from Table 1, we expect:

| Ablation | Effective Depth | Expected Δ | Notes |
|----------|----------------|------------|-------|
| Baseline TRM | 42 | 0.0% | Reference |
| No EMA | 42 | -7.5% | Training instability |
| With ACT | 42 | -1.3% | 2× slower training |
| T=2, n=2 | 12 | -13.7% | Insufficient recursion |
| 4-layers, n=3 | 48 | -7.9% | Overfitting |

## Monitoring

All experiments log to Weights & Biases (W&B). Monitor:
- Training loss
- Test accuracy
- Average halting steps
- Learning rate schedule

## Analysis

After running all experiments, compare:
1. **Final test accuracy** - which ablation hurts performance most?
2. **Training stability** - loss curves, especially for no-EMA
3. **Training speed** - steps/sec, especially for ACT continue
4. **Effective capacity** - does more layers or more recursion work better?

## References

Paper: "Less is More: Recursive Reasoning with Tiny Networks"
- Table 1: Ablation of TRM on Sudoku-Extreme
- Section 4: Tiny Recursion Models methodology


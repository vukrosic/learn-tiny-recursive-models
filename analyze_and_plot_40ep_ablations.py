#!/usr/bin/env python3
"""
Analyze and plot results from 40-epoch ablation experiments.
Creates comparison plots and extracts metrics for the article.
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(checkpoint_dir):
    """Load training metrics from checkpoint directory."""
    # Look for wandb metrics file
    wandb_dirs = glob.glob(os.path.join(checkpoint_dir, "wandb", "latest-run", "files"))
    if not wandb_dirs:
        # Try offline runs
        wandb_dirs = glob.glob(os.path.join(checkpoint_dir, "wandb", "offline-run-*", "files"))
    
    if not wandb_dirs:
        print(f"No wandb directory found in {checkpoint_dir}")
        return None
    
    wandb_dir = wandb_dirs[-1]  # Get most recent
    
    # Try to find metrics file
    history_file = os.path.join(os.path.dirname(wandb_dir), "files", "wandb-history.jsonl")
    if not os.path.exists(history_file):
        print(f"No history file found: {history_file}")
        return None
    
    # Load metrics
    steps = []
    lm_losses = []
    
    with open(history_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if 'train/lm_loss' in data and '_step' in data:
                steps.append(data['_step'])
                lm_losses.append(data['train/lm_loss'])
    
    if not steps:
        print(f"No training metrics found in {history_file}")
        return None
    
    return {
        'steps': steps,
        'lm_losses': lm_losses,
        'initial_loss': lm_losses[0] if lm_losses else None,
        'final_loss': lm_losses[-1] if lm_losses else None,
        'min_loss': min(lm_losses) if lm_losses else None
    }

def main():
    print("=" * 80)
    print("Analyzing 40-Epoch Ablation Results")
    print("=" * 80)
    print()
    
    experiments = {
        'Baseline': 'checkpoints/Maze-30x30-hard-1k-ACT-torch/ablation_baseline_40ep',
        'No EMA': 'checkpoints/Maze-30x30-hard-1k-ACT-torch/ablation_no_ema_40ep',
        'Less Recursion': 'checkpoints/Maze-30x30-hard-1k-ACT-torch/ablation_less_recursion_40ep',
        'Bigger Brain (4-layer)': 'checkpoints/Maze-30x30-hard-1k-ACT-torch/ablation_bigger_brain_40ep'
    }
    
    results = {}
    for name, path in experiments.items():
        print(f"Loading {name}...")
        metrics = load_metrics(path)
        if metrics:
            results[name] = metrics
            print(f"  Initial Loss: {metrics['initial_loss']:.4f}")
            print(f"  Final Loss: {metrics['final_loss']:.4f}")
            print(f"  Min Loss: {metrics['min_loss']:.4f}")
        else:
            print(f"  Could not load metrics")
        print()
    
    if not results:
        print("No results to plot!")
        return
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = {
        'Baseline': '#1f77b4',  # Blue
        'No EMA': '#ff7f0e',     # Orange
        'Less Recursion': '#2ca02c',  # Green
        'Bigger Brain (4-layer)': '#d62728'  # Red
    }
    
    linestyles = {
        'Baseline': '-',
        'No EMA': '--',
        'Less Recursion': '-.',
        'Bigger Brain (4-layer)': ':'
    }
    
    for name, metrics in results.items():
        plt.plot(
            metrics['steps'],
            metrics['lm_losses'],
            label=name,
            color=colors.get(name, None),
            linestyle=linestyles.get(name, '-'),
            linewidth=2
        )
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('LM Loss', fontsize=12)
    plt.title('TRM Ablation Study: Training Loss Comparison (40 Epochs)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = 'docs/images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'complete_ablation_study_40ep.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Save metrics to JSON
    summary = {}
    for name, metrics in results.items():
        summary[name] = {
            'initial_loss': metrics['initial_loss'],
            'final_loss': metrics['final_loss'],
            'min_loss': metrics['min_loss'],
            'improvement_pct': ((metrics['initial_loss'] - metrics['final_loss']) / metrics['initial_loss'] * 100) if metrics['initial_loss'] else 0
        }
    
    summary_path = 'results/ablation_40ep_summary.json'
    os.makedirs('results', exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    # Print summary table
    print()
    print("=" * 80)
    print("Results Summary:")
    print("-" * 80)
    print(f"{'Configuration':<30} {'Initial':>10} {'Final':>10} {'Min':>10} {'Improv %':>10}")
    print("-" * 80)
    for name, data in summary.items():
        print(f"{name:<30} {data['initial_loss']:>10.4f} {data['final_loss']:>10.4f} {data['min_loss']:>10.4f} {data['improvement_pct']:>10.1f}")
    print("=" * 80)

if __name__ == "__main__":
    main()


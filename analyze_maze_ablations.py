#!/usr/bin/env python3
"""
Analyze and compare results from TRM ablation experiments on Maze dataset.

Usage:
    python analyze_maze_ablations.py
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def find_checkpoint_dirs(base_dir: str = "checkpoints") -> Dict[str, str]:
    """Find checkpoint directories for each ablation experiment."""
    experiments = {
        "Baseline TRM": "Maze*TinyRecursive*",
        "No EMA": "Maze*no*ema*",
        "With ACT Continue": "Maze*with*act*",
        "T=2, n=2": "Maze*T2*n2*",
        "4-layers, n=3": "Maze*4layers*"
    }
    
    found_dirs = {}
    for name, pattern in experiments.items():
        matches = glob.glob(os.path.join(base_dir, pattern))
        if matches:
            # Get most recent if multiple matches
            found_dirs[name] = max(matches, key=os.path.getmtime)
    
    return found_dirs


def extract_metrics_from_checkpoint(checkpoint_dir: str) -> Optional[Dict]:
    """Extract key metrics from checkpoint directory."""
    # Look for wandb run directory
    wandb_dirs = glob.glob(os.path.join(checkpoint_dir, "wandb", "latest-run"))
    
    if not wandb_dirs:
        return None
    
    # Try to read summary.json
    summary_file = os.path.join(wandb_dirs[0], "files", "wandb-summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            return {
                "test_accuracy": summary.get("test/accuracy", None),
                "test_loss": summary.get("test/loss", None),
                "train_loss": summary.get("train/loss", None),
                "avg_steps": summary.get("train/avg_steps", None),
            }
    
    return None


def count_parameters(config_name: str) -> int:
    """Estimate number of parameters based on config."""
    # Read config
    config_path = f"config/{config_name}.yaml"
    if not os.path.exists(config_path):
        return 0
    
    # For TRM with hidden_size=512, 2-layers, this is approximately 5-7M params
    # This is a rough estimate
    estimates = {
        "cfg_pretrain_maze": 7_000_000,
        "cfg_pretrain_maze_no_ema": 7_000_000,
        "cfg_pretrain_maze_with_act": 7_000_000,
        "cfg_pretrain_maze_T2_n2": 7_000_000,
        "cfg_pretrain_maze_4layers": 14_000_000,  # ~2x for 4 layers
    }
    
    return estimates.get(config_name, 7_000_000)


def calculate_effective_depth(T: int, n: int, layers: int) -> int:
    """Calculate effective depth: T × (n+1) × layers"""
    return T * (n + 1) * layers


def create_comparison_table() -> pd.DataFrame:
    """Create comparison table with experiment details."""
    
    ablations = [
        {
            "Experiment": "Baseline TRM",
            "T": 3,
            "n": 6,
            "Layers": 2,
            "Effective Depth": calculate_effective_depth(3, 6, 2),
            "EMA": "Yes",
            "ACT Continue": "No",
            "Params (M)": "~7",
            "Config": "cfg_pretrain_maze"
        },
        {
            "Experiment": "No EMA",
            "T": 3,
            "n": 6,
            "Layers": 2,
            "Effective Depth": calculate_effective_depth(3, 6, 2),
            "EMA": "No",
            "ACT Continue": "No",
            "Params (M)": "~7",
            "Config": "cfg_pretrain_maze_no_ema"
        },
        {
            "Experiment": "With ACT Continue",
            "T": 3,
            "n": 6,
            "Layers": 2,
            "Effective Depth": calculate_effective_depth(3, 6, 2),
            "EMA": "Yes",
            "ACT Continue": "Yes",
            "Params (M)": "~7",
            "Config": "cfg_pretrain_maze_with_act"
        },
        {
            "Experiment": "T=2, n=2",
            "T": 2,
            "n": 2,
            "Layers": 2,
            "Effective Depth": calculate_effective_depth(2, 2, 2),
            "EMA": "Yes",
            "ACT Continue": "No",
            "Params (M)": "~7",
            "Config": "cfg_pretrain_maze_T2_n2"
        },
        {
            "Experiment": "4-layers, n=3",
            "T": 3,
            "n": 3,
            "Layers": 4,
            "Effective Depth": calculate_effective_depth(3, 3, 4),
            "EMA": "Yes",
            "ACT Continue": "No",
            "Params (M)": "~14",
            "Config": "cfg_pretrain_maze_4layers"
        },
    ]
    
    return pd.DataFrame(ablations)


def main():
    print("=" * 80)
    print("TRM Ablation Study - Maze Dataset Results")
    print("=" * 80)
    print()
    
    # Create configuration table
    print("Experiment Configurations:")
    print("-" * 80)
    config_df = create_comparison_table()
    print(config_df.to_string(index=False))
    print()
    print()
    
    # Find checkpoint directories
    print("Searching for checkpoint directories...")
    checkpoint_dirs = find_checkpoint_dirs()
    
    if not checkpoint_dirs:
        print("No checkpoint directories found. Have you run the experiments yet?")
        print()
        print("Run experiments with:")
        print("  bash run_maze_ablations.sh")
        return
    
    print(f"Found {len(checkpoint_dirs)} experiment(s)")
    print()
    
    # Extract metrics
    results = []
    for name, checkpoint_dir in checkpoint_dirs.items():
        print(f"Analyzing: {name}")
        print(f"  Directory: {checkpoint_dir}")
        
        metrics = extract_metrics_from_checkpoint(checkpoint_dir)
        if metrics:
            results.append({
                "Experiment": name,
                **metrics
            })
            print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
            print(f"  Test Loss: {metrics.get('test_loss', 'N/A')}")
        else:
            print("  Could not extract metrics (experiment may still be running)")
        print()
    
    # Create results table
    if results:
        print()
        print("=" * 80)
        print("Results Summary:")
        print("-" * 80)
        results_df = pd.DataFrame(results)
        
        # Merge with config info
        config_df_subset = config_df[["Experiment", "T", "n", "Layers", "Effective Depth"]]
        merged_df = pd.merge(config_df_subset, results_df, on="Experiment", how="left")
        
        # Calculate relative performance
        if "test_accuracy" in merged_df.columns:
            baseline_acc = merged_df[merged_df["Experiment"] == "Baseline TRM"]["test_accuracy"].values
            if len(baseline_acc) > 0 and baseline_acc[0] is not None:
                merged_df["Δ from Baseline (%)"] = merged_df["test_accuracy"].apply(
                    lambda x: f"{(x - baseline_acc[0]) * 100:.1f}" if x is not None else "N/A"
                )
        
        print(merged_df.to_string(index=False))
        print()
        
        # Save to CSV
        output_file = "maze_ablation_results.csv"
        merged_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        print()
    
    print("=" * 80)
    print("Analysis complete!")
    print()
    print("For detailed metrics, check your Weights & Biases dashboard")
    print("=" * 80)


if __name__ == "__main__":
    main()


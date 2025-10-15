#!/usr/bin/env python3
"""Generate loss plot from training_metrics.json"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load metrics
with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Extract data
steps = []
lm_losses = []
q_halt_losses = []

for m in metrics:
    if 'step' in m and 'train/lm_loss' in m:
        steps.append(m['step'])
        lm_losses.append(m['train/lm_loss'])
        if 'train/q_halt_loss' in m:
            q_halt_losses.append(m['train/q_halt_loss'])

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot LM loss
ax1.plot(steps, lm_losses, 'b-', linewidth=2, marker='o', markersize=3)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('LM Loss', fontsize=12)
ax1.set_title('Language Model Loss Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot Q halt loss
if q_halt_losses:
    ax2.plot(steps, q_halt_losses, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Q Halt Loss', fontsize=12)
    ax2.set_title('Q Halt Loss Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to training_loss.png")

# Print summary
print(f"\n{'='*50}")
print(f"Loss Summary:")
print(f"  Initial LM loss: {lm_losses[0]:.6f}")
print(f"  Final LM loss: {lm_losses[-1]:.6f}")
print(f"  Min LM loss: {min(lm_losses):.6f}")
print(f"  Max LM loss: {max(lm_losses):.6f}")
print(f"  Total steps: {len(steps)}")
print(f"{'='*50}")


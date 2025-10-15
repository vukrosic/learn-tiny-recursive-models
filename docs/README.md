# Documentation Assets

This directory contains documentation images and figures that are referenced in the main documentation files.

## Images

### Ablation Study Results

- **`complete_ablation_study.png`**: Complete comparison of all 4 ablation configurations
  - Used in `ARTICLE.md` to illustrate experimental results
  - Shows LM loss curves for: Baseline, No EMA, Less Recursion, and Bigger Brain configurations
  - Generated from 10-epoch training runs on maze-solving task

## Why This Directory Exists

Unlike the `results/` directory (which is gitignored and contains user-generated experimental outputs), this `docs/` directory is **committed to version control** to ensure that documentation images are available to all users of the repository.

When you clone this repository, these images will be included so that documentation files render correctly.

## Updating Images

If you regenerate ablation study results and want to update the documentation images:

1. Run the ablation experiments (see `results/README.md`)
2. Copy the desired plot from `results/plots/` to `docs/images/`
3. Commit the updated image

Example:
```bash
cp results/plots/complete_ablation_study.png docs/images/
git add docs/images/complete_ablation_study.png
git commit -m "Update ablation study figure"
```


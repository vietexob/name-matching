# Pipeline Script Usage Examples

This document provides common usage examples for the `train_pipeline.sh` script.

## Quick Start

### Run Full Pipeline with Defaults
```bash
./train_pipeline.sh
```
This will:
- Generate 700 person names and 300 organization names
- Create training pairs with 10 negative examples per positive
- Train model with test_size=0.2 and threshold=0.85

### View Help and All Options
```bash
./train_pipeline.sh --help
```

## Common Scenarios

### 1. Training with More Data
Generate more synthetic names for better model performance:
```bash
./train_pipeline.sh --n-persons 1500 --n-orgas 800
```

### 2. Higher Precision Model
Use a higher classification threshold for fewer false positives:
```bash
./train_pipeline.sh --thresh 0.9 --human-readable
```

### 3. More Balanced Dataset
Increase negative examples to balance the dataset:
```bash
./train_pipeline.sh --n-neg 15
```

### 4. Hyperparameter Tuning
Enable Optuna-based hyperparameter optimization:
```bash
# Quick tuning (25 trials, ~10-15 minutes extra)
./train_pipeline.sh --tune

# Extensive tuning (100 trials, ~30-60 minutes extra)
./train_pipeline.sh --tune --n-trials 100
```

### 5. Skip Completed Steps
If you've already generated data and only want to retrain:
```bash
# Skip name generation and dataset creation
# Only train model (if pairs already exist)
./train_pipeline.sh --skip-generate --skip-dataset

# Only generate new training pairs and train
./train_pipeline.sh --skip-generate
```

### 6. Quiet Mode
Run with minimal script output (Python logs still visible):
```bash
./train_pipeline.sh --quiet
```

### 7. Production-Ready Configuration
Best practices for production model:
```bash
./train_pipeline.sh \
  --n-persons 2000 \
  --n-orgas 1000 \
  --n-neg 15 \
  --test-size 0.25 \
  --thresh 0.88 \
  --tune \
  --n-trials 50 \
  --human-readable
```

## Step-by-Step Workflow

### Initial Development
```bash
# 1. Quick test with small dataset
./train_pipeline.sh --n-persons 100 --n-orgas 50 --n-neg 5

# 2. Review results, then scale up
./train_pipeline.sh --n-persons 1000 --n-orgas 500

# 3. Fine-tune threshold based on results
./train_pipeline.sh --skip-generate --skip-dataset --thresh 0.87
```

### Retraining After Data Changes
```bash
# If you added new names to data/raw/mock_sample.csv:
./train_pipeline.sh --skip-generate

# If you modified training pairs directly:
./train_pipeline.sh --skip-generate --skip-dataset
```

### Experimenting with Thresholds
```bash
# Test different thresholds without regenerating data:
./train_pipeline.sh --skip-generate --skip-dataset --thresh 0.80
./train_pipeline.sh --skip-generate --skip-dataset --thresh 0.85
./train_pipeline.sh --skip-generate --skip-dataset --thresh 0.90
```

## Combining Options

### Fast Iteration (Minimal Logging)
```bash
./train_pipeline.sh --silent --disable-tqdm --quiet
```

### Detailed Debugging (Maximum Output)
```bash
./train_pipeline.sh --human-readable
```

### Large-Scale Production Training
```bash
./train_pipeline.sh \
  --n-persons 5000 \
  --n-orgas 2000 \
  --n-neg 20 \
  --test-size 0.2 \
  --tune \
  --n-trials 100 \
  --human-readable \
  --silent
```

## Troubleshooting

### Azure OpenAI Errors
If you get authentication errors during dataset generation:
```bash
# Check your .env file has all required variables
cat .env

# Skip dataset generation and use existing pairs
./train_pipeline.sh --skip-dataset
```

### Out of Memory
If training fails due to memory issues:
```bash
# Reduce dataset size
./train_pipeline.sh --n-persons 500 --n-orgas 200 --n-neg 5

# Or skip name generation and manually reduce existing data
```

### Slow Performance
If the pipeline is too slow:
```bash
# Disable progress bars
./train_pipeline.sh --disable-tqdm

# Reduce dataset size
./train_pipeline.sh --n-persons 300 --n-orgas 150 --n-neg 5

# Skip hyperparameter tuning
./train_pipeline.sh --tune false
```

## Output Files

After successful execution, you'll have:

```
data/
├── raw/
│   └── mock_sample.csv                          # Synthetic names
└── processed/
    ├── name_matching_pos_pairs.csv             # Positive pairs
    ├── name_matching_neg_pairs.csv             # Negative pairs
    └── name_matching_training_featured.csv     # Featurized data

models/
├── model_lgb_name_matching.pkl                 # Trained model
└── name_matching_tfidf_ngrams.pkl              # TF-IDF vectorizer

reports/
└── figures/
    ├── roc_auc_train.png                       # ROC curve
    ├── precision_recall_curve.png              # PR curve
    ├── feature_importance.png                  # Feature importance
    └── feature_distribution.png                # Feature distributions
```

## Next Steps After Pipeline

### 1. Test the Model
```bash
pytest tests/ -v
```

### 2. Run the API
```bash
python app.py
```

### 3. Perform Entity Resolution
```bash
python entity_resolution.py
```

### 4. Interactive Testing
```bash
python example_api_usage.py
```

## Advanced Usage

### Custom Configuration Files
The scripts read from `name_matching/config/Config.ini`. Modify this file to change paths, column names, or other settings.

### Using Real Data
Instead of synthetic names, place your own data in `data/raw/mock_sample.csv`:
```bash
# Skip synthetic generation, use your real data
./train_pipeline.sh --skip-generate
```

Required columns: `FULL_NAME`, `FIRST_NAME`, `LAST_NAME`, `ENT_TYPE`

### Parallel Experiments
Run multiple configurations in parallel using background jobs:
```bash
./train_pipeline.sh --thresh 0.80 > logs/run_080.log 2>&1 &
./train_pipeline.sh --thresh 0.85 > logs/run_085.log 2>&1 &
./train_pipeline.sh --thresh 0.90 > logs/run_090.log 2>&1 &
```

## Performance Tips

1. **Start Small**: Test with `--n-persons 100` first
2. **Incremental Training**: Use `--skip-*` flags to avoid regenerating data
3. **Tune Last**: Only use `--tune` after you're satisfied with the base model
4. **Monitor Resources**: Watch memory usage during training with large datasets
5. **Save Logs**: Redirect output to files for later analysis: `./train_pipeline.sh > pipeline.log 2>&1`

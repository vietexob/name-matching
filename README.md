## Project Overview

This is a Name Matching ML model for entity resolution and transaction monitoring. It uses a LightGBM classifier to determine whether two names (person or organization) refer to the same entity. The model uses multiple string similarity features including edit distance, Jaccard similarity, TF-IDF cosine similarity, and sentence embeddings.

## Architecture

### Three-Stage Pipeline

The codebase implements a three-stage ML pipeline:

1. **Data Generation** (`name_matching/data/`)
   - `generate_names.py`: Creates synthetic names using `Faker` library for Western/Asian persons and organizations
   - `make_dataset.py`: Generates training pairs (positive and negative examples) using Azure OpenAI to create aliases

2. **Feature Engineering** (`name_matching/features/`)
   - `build_features.py`: Computes 7 similarity features between name pairs:
     - Jaccard similarity (token intersection over union)
     - TF-IDF cosine similarity (requires pre-fitted vectorizer)
     - Ratio feature (normalized edit distance)
     - Sorted token ratio (edit distance on sorted tokens)
     - Token set ratio (edit distance on unique sorted tokens)
     - Partial ratio (fuzzy matching)
     - Embedding distance (sentence-transformers: all-MiniLM-L6-v2)

3. **Model Training** (`name_matching/models/`)
   - `train_model.py`: Trains LightGBM classifier, generates performance plots, saves model and TF-IDF vectorizer as pickle files

### Key Components

- **Configuration**: All paths and column names centralized in `name_matching/config/Config.ini`
- **Logging**: Uses structlog throughout, configured via CLI flags (`--silent`, `--human-readable`)
- **Entity Types**: Handles both "PERS" (person) and "ORGA" (organization) entities differently

### Data Flow

```
Raw names → Alias generation (Azure OpenAI) → Positive/Negative pairs →
Feature engineering (TF-IDF + embeddings) → LightGBM training →
Saved models (model_lgb_name_matching.pkl, name_matching_tfidf_ngrams.pkl)
```

## Common Commands

### Generate Synthetic Training Data
```bash
python -m name_matching.data.generate_names --n_persons 700 --n_orgas 300
```

### Create Training Pairs
```bash
python -m name_matching.data.make_dataset --n_neg 10
```
- Requires Azure OpenAI credentials in `.env` file
- Generates positive pairs via LLM-generated aliases and typos
- Generates negative pairs using edit distance to find hard negatives
- Output: `data/processed/name_matching_pos_pairs.csv` and `name_matching_neg_pairs.csv`

### Train Model
```bash
python -m name_matching.models.train_model --test-size 0.2 --thresh 0.85 --human-readable
```
- Loads positive/negative pairs, builds features, trains LightGBM classifier
- Output: Model pickles in `models/` and performance plots in `reports/figures/`
- `--thresh`: Classification threshold for positive class prediction

### CLI Options (All Scripts)
- `-s, --silent`: Enable INFO logging (default behavior, less verbose)
- `-hr, --human-readable`: Pretty-print classification report
- `-dt, --disable-tqdm`: Disable progress bars

## Installation

Install all dependencies:
```bash
pip install -r requirements.txt
```

Download required NLTK data (for stopwords):
```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Environment Setup

Required environment variables in `.env`:
```
OPENAI_API_KEY=<legacy-key>
AZURE_OPENAI_API_VERSION=<version>
AZURE_OPENAI_DEPLOYMENT=<model-name>
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=<endpoint-url>
```

## Important Implementation Details

### TF-IDF Vectorizer Dependency
The feature generation requires a pre-fitted TF-IDF vectorizer. During training, `FeatureGenerator.create_tfidf_vectorizer()` creates and saves it. For inference, the saved vectorizer must be loaded first.

### Name Normalization
Names are normalized in `make_dataset.py` via `process_text_standard()`:
- Convert to uppercase
- Remove special characters and punctuation
- Optionally remove stopwords (disabled for names)
- Numeric token removal configurable

### Negative Sampling Strategy
Hard negative mining in `TrainingDataGenerator.generate_neg_mappings()`:
1. Sample candidates with same first/last name (confusable pairs)
2. Sort remaining by edit distance
3. Select top `n` closest non-matches per positive example

### Model Artifacts
All models and data stored in directories specified by Config.ini:
- `models/`: Trained model and vectorizer pickles
- `data/raw/`: Synthetic or uploaded raw name lists
- `data/processed/`: Generated positive/negative pairs
- `reports/figures/`: ROC-AUC, feature importance, and PR curves

## Running the Full Training Pipeline

### Automated Pipeline (Recommended)

Use the provided bash script to run the entire pipeline automatically:

```bash
# Run full pipeline with defaults:
./train_pipeline.sh

# Customize parameters:
./train_pipeline.sh --n-persons 1000 --n-orgas 500 --thresh 0.9 --human-readable

# Enable hyperparameter tuning:
./train_pipeline.sh --tune --n-trials 50

# Skip certain steps (if already completed):
./train_pipeline.sh --skip-generate --skip-dataset

# See all options:
./train_pipeline.sh --help
```

The script provides:
- Color-coded output for easy progress tracking
- Prerequisite checks (Python, .env file, directories)
- Error handling and validation
- Progress summaries and file counts
- Total runtime reporting
- Ability to skip completed steps

### Manual Pipeline Execution

Execute steps individually in order:
```bash
# 1. Generate synthetic names (optional if you have real data)
python -m name_matching.data.generate_names --n_persons 700 --n_orgas 300

# 2. Generate training pairs (requires Azure OpenAI)
python -m name_matching.data.make_dataset --n_neg 10

# 3. Train the model
python -m name_matching.models.train_model --test-size 0.2 --thresh 0.85 --human-readable
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Unit Tests Only
```bash
pytest tests/unit_tests/ -v
```

### Run Integration Tests (API Tests)
```bash
pytest tests/integration_tests/test_api.py -v
```

### Run Specific Test File
```bash
pytest tests/unit_tests/test_predict_model.py -v
```

## API Usage

### Running the Flask API

Start the development server:
```bash
python app.py
```
The API will run on `http://localhost:5001`

For production deployment with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### API Endpoints

- `GET /health` - Health check
- `GET /info` - Model information
- `POST /predict` - Single name pair prediction
- `POST /predict/batch` - Batch predictions

### Example API Usage

See `example_api_usage.py` for comprehensive examples. Basic usage:

```bash
# In terminal 1: Start the API
python app.py

# In terminal 2: Run examples
python example_api_usage.py
```

Or make direct requests:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CUST_NAME": "John Smith",
    "COUNTERPART_NAME": "J. Smith",
    "FT_NO": "FT12345",
    "threshold": 0.85
  }'
```

## Entity Resolution

The `entity_resolution.py` script implements a complete entity resolution pipeline using graph-based community detection:

```bash
python entity_resolution.py
```

### Entity Resolution Architecture

1. **Load & Preprocess**: Loads transaction data, normalizes names using `process_text_standard()`
2. **Deduplication**: Removes duplicate name pairs
3. **Pairwise Comparison**: Generates all unique pairwise combinations (`n` choose 2 pairs)
4. **Batch Prediction**: Uses trained model to predict matches for all pairs
5. **Graph Construction**: Creates NetworkX graph where edges represent matched pairs
6. **Community Detection**: Applies Louvain algorithm to find entity clusters
7. **Entity Assignment**: Maps original transaction names to resolved entity IDs
8. **Visualization**: Generates before/after graph visualizations

**Input**: CSV file with `Cust_Name` and `Counterpart_Name` columns
**Output**:
- `data/processed/resolved_txns.csv` with `ENTITY_X`, `ENTITY_Y`, `RESOLVED_NAME_X`, `RESOLVED_NAME_Y` columns
- Graph visualizations in `reports/figures/`

### Entity Resolution vs. Model Training

- **Model Training** (`train_model.py`): Trains the LightGBM classifier on labeled pairs
- **Entity Resolution** (`entity_resolution.py`): Uses the trained model to resolve entities in unlabeled transaction data via graph algorithms

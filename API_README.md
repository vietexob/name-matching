# Name Matching API Documentation

A Flask-based REST API for real-time name matching classification for entity resolution and transaction monitoring.

## Overview

The Name Matching API provides endpoints for classifying whether two names refer to the same entity. It uses a trained LightGBM classifier with multiple string similarity features including edit distance, Jaccard similarity, TF-IDF cosine similarity, and sentence embeddings.

## Prerequisites

Before running the API, ensure you have:

1. Trained the model by running the full pipeline:
   ```bash
   python -m name_matching.data.generate_names --n_persons 700 --n_orgas 300
   python -m name_matching.data.make_dataset --n_neg 10
   python -m name_matching.models.train_model --test-size 0.2 --thresh 0.85
   ```

2. Model files exist at:
   - `models/model_lgb_name_matching.pkl`
   - `models/name_matching_tfidf_ngrams.pkl`

## Installation

Install the required dependencies:

```bash
pip install flask
```

All other dependencies should already be installed from the main project requirements.

## Running the API

### Development Mode

```bash
python app.py
```

The API will start on `http://localhost:5001`

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

Options:
- `-w 4`: Number of worker processes (adjust based on CPU cores)
- `-b 0.0.0.0:5001`: Bind to all interfaces on port 5001
- `--timeout 120`: Request timeout in seconds

## API Endpoints

### 1. Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "name-matching-api"
}
```

---

### 2. Model Information

Get information about the loaded model.

**Endpoint:** `GET /info`

**Response:**
```json
{
  "status": "success",
  "model": {
    "type": "LightGBM Classifier",
    "model_path": "models/model_lgb_name_matching.pkl",
    "tfidf_path": "models/name_matching_tfidf_ngrams.pkl",
    "features": [
      "JACCARD_SIM",
      "COSINE_SIM",
      "RATIO",
      "SORTED_TOKEN_RATIO",
      "TOKEN_SET_RATIO",
      "PARTIAL_RATIO",
      "EMB_DISTANCE"
    ],
    "num_features": 7
  }
}
```

---

### 3. Single Prediction

Classify a single name pair.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "CUST_NAME": "John Smith",
  "COUNTERPART_NAME": "J. Smith",
  "FT_NO": "FT12345",
  "threshold": 0.85
}
```

**Parameters:**
- `CUST_NAME` (required): First name (customer name)
- `COUNTERPART_NAME` (required): Second name (counterpart name)
- `FT_NO` (optional): Transaction reference number for tracking
- `threshold` (optional): Classification threshold (default: 0.85)

**Response:**
```json
{
  "status": "success",
  "result": {
    "ft_no": "FT12345",
    "name_x": "John Smith",
    "name_y": "J. Smith",
    "prediction": 1,
    "match_label": "MATCH",
    "probability": 0.9234,
    "threshold": 0.85,
    "features": {
      "JACCARD_SIM": 0.5,
      "COSINE_SIM": 0.8765,
      "RATIO": 0.7143,
      "SORTED_TOKEN_RATIO": 0.7143,
      "TOKEN_SET_RATIO": 0.7143,
      "PARTIAL_RATIO": 0.8,
      "EMB_DISTANCE": 0.9123
    }
  }
}
```

**Response Fields:**
- `prediction`: Binary prediction (0 = no match, 1 = match)
- `match_label`: Human-readable label ("MATCH" or "NO_MATCH")
- `probability`: Confidence score (0-1)
- `features`: Individual feature values used for prediction

---

### 4. Batch Prediction

Classify multiple name pairs in a single request.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "pairs": [
    {
      "CUST_NAME": "John Smith",
      "COUNTERPART_NAME": "J. Smith",
      "FT_NO": "FT001"
    },
    {
      "CUST_NAME": "Apple Inc.",
      "COUNTERPART_NAME": "Apple Corporation",
      "FT_NO": "FT002"
    }
  ],
  "threshold": 0.85
}
```

**Parameters:**
- `pairs` (required): List of name pair objects
- `threshold` (optional): Classification threshold for all pairs (default: 0.85)

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "results": [
    {
      "ft_no": "FT001",
      "name_x": "John Smith",
      "name_y": "J. Smith",
      "prediction": 1,
      "match_label": "MATCH",
      "probability": 0.9234,
      "threshold": 0.85,
      "features": { ... }
    },
    {
      "ft_no": "FT002",
      "name_x": "Apple Inc.",
      "name_y": "Apple Corporation",
      "prediction": 1,
      "match_label": "MATCH",
      "probability": 0.8876,
      "threshold": 0.85,
      "features": { ... }
    }
  ]
}
```

**Partial Success (207 Multi-Status):**

If some predictions fail, the API returns a 207 status:

```json
{
  "status": "partial_success",
  "message": "1 out of 2 predictions failed",
  "results": [
    { ... },
    {
      "error": "Validation error",
      "message": "Both name_x and name_y must be non-empty strings",
      "ft_no": "FT002"
    }
  ]
}
```

---

## Usage Examples

### Python with requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:5001/predict",
    json={
        "CUST_NAME": "John Smith",
        "COUNTERPART_NAME": "J. Smith",
        "FT_NO": "FT12345",
        "threshold": 0.85
    }
)
result = response.json()
print(f"Match: {result['result']['match_label']}")
print(f"Probability: {result['result']['probability']}")

# Batch prediction
response = requests.post(
    "http://localhost:5001/predict/batch",
    json={
        "pairs": [
            {"CUST_NAME": "John Doe", "COUNTERPART_NAME": "J. Doe", "FT_NO": "FT001"},
            {"CUST_NAME": "Apple Inc", "COUNTERPART_NAME": "Apple Corp", "FT_NO": "FT002"}
        ],
        "threshold": 0.85
    }
)
results = response.json()
for result in results['results']:
    print(f"{result['ft_no']}: {result['match_label']} ({result['probability']:.2%})")
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CUST_NAME": "John Smith",
    "COUNTERPART_NAME": "J. Smith",
    "FT_NO": "FT12345",
    "threshold": 0.85
  }'

# Batch prediction
curl -X POST http://localhost:5001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {"CUST_NAME": "John Doe", "COUNTERPART_NAME": "J. Doe", "FT_NO": "FT001"},
      {"CUST_NAME": "Apple Inc", "COUNTERPART_NAME": "Apple Corp", "FT_NO": "FT002"}
    ],
    "threshold": 0.85
  }'
```

### JavaScript (fetch)

```javascript
// Single prediction
const response = await fetch('http://localhost:5001/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    CUST_NAME: 'John Smith',
    COUNTERPART_NAME: 'J. Smith',
    FT_NO: 'FT12345',
    threshold: 0.85
  })
});
const result = await response.json();
console.log(`Match: ${result.result.match_label}`);
```

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "message": "Error description",
  "ft_no": "FT12345"
}
```

### Common Error Codes

- **400 Bad Request**: Invalid input data
  - Missing required fields
  - Invalid data types
  - Invalid threshold value

- **404 Not Found**: Endpoint does not exist

- **405 Method Not Allowed**: Wrong HTTP method used

- **500 Internal Server Error**: Server-side error
  - Model loading failure
  - Feature generation error
  - Prediction error

---

## Testing

### Run Unit Tests

Test the prediction module:

```bash
pytest tests/unit_tests/test_predict_model.py -v
```

### Run Integration Tests

Test the API endpoints:

```bash
pytest tests/integration_tests/test_api.py -v
```

### Run All Tests

```bash
pytest tests/ -v
```

---

## Performance Considerations

1. **Model Loading**: The model is loaded once at startup and cached for subsequent requests.

2. **Batch Processing**: Use the `/predict/batch` endpoint for multiple predictions to reduce overhead.

3. **Threshold Tuning**: Adjust the threshold based on your use case:
   - Lower threshold (e.g., 0.5): Higher recall, more false positives
   - Higher threshold (e.g., 0.9): Higher precision, more false negatives

4. **Concurrency**: Use a WSGI server with multiple workers for production deployments.

---

## Configuration

Model paths and column names are configured in `name_matching/config/Config.ini`. To use different model paths, update:

```ini
[MODELPATH]
MODEL_LGB_NAME_MATCHING = models/model_lgb_name_matching.pkl
FILENAME_MODEL_TFIDF_NGRAM = models/name_matching_tfidf_ngrams.pkl
```

---

## Monitoring and Logging

The API uses structlog for structured logging. All predictions and errors are logged with relevant metadata:

- Request information (method, path, remote address)
- Prediction requests (names, transaction reference)
- Prediction results (match label, probability)
- Errors (validation, type, prediction errors)

---

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model file not found
```

**Solution**: Train the model first using:
```bash
python -m name_matching.models.train_model
```

### Empty Prediction Response

**Solution**: Check that both names are non-empty strings and properly formatted in the request.

### Low Prediction Accuracy

**Solution**:
1. Retrain the model with more training data
2. Adjust the classification threshold
3. Check that input names are properly normalized

---

## Support

For issues or questions:
- Check the main project documentation in `CLAUDE.md`
- Review test cases in `tests/` for usage examples
- Ensure all model files exist and are properly trained

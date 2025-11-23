"""
Flask API for Name Matching Classification

This API provides real-time name matching classification for entity resolution
and transaction monitoring.

Endpoints:
    - POST /predict: Single name pair classification
    - POST /predict/batch: Batch name pair classification
    - GET /health: Health check endpoint
    - GET /info: Model information endpoint
"""

import os
from typing import Dict

from flask import Flask, jsonify, request

from name_matching.log.logging import configure_structlog
from name_matching.models.predict_model import NameMatchingPredictor

# Disable tokenizer parallelism in production
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__)

# Configure logging
configure_structlog(silent=False)

# Initialize predictor (singleton pattern)
predictor = None


def get_predictor() -> NameMatchingPredictor:
    """
    Gets or creates the predictor instance (singleton pattern).

    :return: NameMatchingPredictor instance
    """
    global predictor
    if predictor is None:
        predictor = NameMatchingPredictor()
    return predictor


@app.before_request
def log_request_info():
    """Logs incoming request information."""
    app.logger.info(
        "INCOMING_REQUEST",
        method=request.method,
        path=request.path,
        remote_addr=request.remote_addr,
    )


@app.route("/health", methods=["GET"])
def health_check() -> Dict:
    """
    Health check endpoint to verify API is running.

    Returns:
        JSON response with status
    """
    return jsonify({"status": "healthy", "service": "name-matching-api"}), 200


@app.route("/info", methods=["GET"])
def model_info() -> Dict:
    """
    Returns information about the loaded model.

    Returns:
        JSON response with model metadata
    """
    try:
        pred = get_predictor()
        return (
            jsonify(
                {
                    "status": "success",
                    "model": {
                        "type": "LightGBM Classifier",
                        "model_path": pred.model_path,
                        "tfidf_path": pred.tfidf_path,
                        "features": pred.features_final,
                        "num_features": len(pred.features_final),
                    },
                }
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Failed to load model info: {e}"}),
            500,
        )


@app.route("/predict", methods=["POST"])
def predict() -> Dict:
    """
    Predicts whether two names refer to the same entity.

    Expected JSON payload:
        {
            "CUST_NAME": "John Smith",
            "COUNTERPART_NAME": "J. Smith",
            "FT_NO": "FT12345",  # Optional
            "threshold": 0.85    # Optional, default: 0.85
        }

    Returns:
        JSON response with prediction results
    """
    try:
        # Parse request data (silent=True to avoid BadRequest exception)
        data = request.get_json(silent=True)

        if not data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "No JSON data provided in request body",
                    }
                ),
                400,
            )

        # Extract required fields
        cust_name = data.get("CUST_NAME", "")
        counterpart_name = data.get("COUNTERPART_NAME", "")
        ft_no = data.get("FT_NO", "")
        threshold = data.get("threshold", 0.85)

        # Validate inputs
        if not cust_name or not counterpart_name:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Both CUST_NAME and COUNTERPART_NAME are required",
                    }
                ),
                400,
            )

        # Validate threshold
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "threshold must be a number between 0 and 1",
                    }
                ),
                400,
            )

        # Get predictor and make prediction
        pred = get_predictor()
        result = pred.predict(
            name_x=cust_name, name_y=counterpart_name, ft_no=ft_no, threshold=threshold
        )

        # Check if prediction was successful
        if "error" in result:
            return jsonify({"status": "error", **result}), 500

        return jsonify({"status": "success", "result": result}), 200

    except Exception as e:
        app.logger.error(f"PREDICTION_ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch() -> Dict:
    """
    Predicts matches for multiple name pairs.

    Expected JSON payload:
        {
            "pairs": [
                {
                    "CUST_NAME": "John Smith",
                    "COUNTERPART_NAME": "J. Smith",
                    "FT_NO": "FT001"  # Optional
                },
                {
                    "CUST_NAME": "Apple Inc",
                    "COUNTERPART_NAME": "Apple Corp",
                    "FT_NO": "FT002"
                }
            ],
            "threshold": 0.85  # Optional, default: 0.85
        }

    Returns:
        JSON response with batch prediction results
    """
    try:
        # Parse request data (silent=True to avoid BadRequest exception)
        data = request.get_json(silent=True)

        if not data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "No JSON data provided in request body",
                    }
                ),
                400,
            )

        # Extract pairs and threshold
        pairs = data.get("pairs", [])
        threshold = data.get("threshold", 0.85)

        # Validate inputs
        if not pairs or not isinstance(pairs, list):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "pairs must be a non-empty list of name pair objects",
                    }
                ),
                400,
            )

        # Validate threshold
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "threshold must be a number between 0 and 1",
                    }
                ),
                400,
            )

        # First pass: Check for format errors (reject entire batch if found)
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Item at index {i} must be a dictionary",
                        }
                    ),
                    400,
                )

        # Second pass: Transform pairs and collect data validation errors
        transformed_pairs = []
        validation_errors = []

        for i, pair in enumerate(pairs):
            cust_name = pair.get("CUST_NAME", "")
            counterpart_name = pair.get("COUNTERPART_NAME", "")
            ft_no = pair.get("FT_NO", "")

            # Check for data validation errors (empty names)
            if not cust_name or not counterpart_name:
                validation_errors.append(
                    {
                        "index": i,
                        "error": f"Both CUST_NAME and COUNTERPART_NAME are required",
                        "ft_no": ft_no if ft_no else None,
                    }
                )
                transformed_pairs.append(None)  # Placeholder for failed validation
                continue

            transformed_pairs.append(
                {"name_x": cust_name, "name_y": counterpart_name, "ft_no": ft_no}
            )

        # If all pairs failed data validation, return error
        if len(validation_errors) == len(pairs):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "All pairs failed validation",
                        "validation_errors": validation_errors,
                    }
                ),
                400,
            )

        # Get predictor and process pairs
        pred = get_predictor()
        results = []

        for i, pair_data in enumerate(transformed_pairs):
            if pair_data is None:
                # This pair failed validation, add error to results
                error_info = next(
                    (e for e in validation_errors if e["index"] == i), None
                )
                # Safely extract names from pairs[i] if it's a dict, otherwise use empty strings
                pair = pairs[i] if isinstance(pairs[i], dict) else {}
                results.append(
                    {
                        "error": error_info["error"]
                        if error_info
                        else "Validation failed",
                        "ft_no": error_info["ft_no"] if error_info else None,
                        "name_x": pair.get("CUST_NAME", ""),
                        "name_y": pair.get("COUNTERPART_NAME", ""),
                    }
                )
            else:
                # Make prediction for valid pair
                try:
                    result = pred.predict(
                        name_x=pair_data["name_x"],
                        name_y=pair_data["name_y"],
                        ft_no=pair_data["ft_no"],
                        threshold=threshold,
                    )
                    results.append(result)
                except Exception as e:
                    # Handle prediction errors
                    results.append(
                        {
                            "error": str(e),
                            "ft_no": pair_data["ft_no"],
                            "name_x": pair_data["name_x"],
                            "name_y": pair_data["name_y"],
                        }
                    )

        # Check for errors in results
        errors = [r for r in results if "error" in r]
        if errors:
            return (
                jsonify(
                    {
                        "status": "partial_success",
                        "message": f"{len(errors)} out of {len(results)} predictions failed",
                        "count": len(results),
                        "results": results,
                    }
                ),
                207,  # Multi-Status
            )

        return (
            jsonify(
                {
                    "status": "success",
                    "count": len(results),
                    "results": results,
                }
            ),
            200,
        )

    except Exception as e:
        app.logger.error(f"BATCH_PREDICTION_ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handles 404 errors."""
    return (
        jsonify(
            {
                "status": "error",
                "message": "Endpoint not found",
                "available_endpoints": [
                    "GET /health",
                    "GET /info",
                    "POST /predict",
                    "POST /predict/batch",
                ],
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handles 500 errors."""
    return (
        jsonify({"status": "error", "message": "Internal server error occurred"}),
        500,
    )


if __name__ == "__main__":
    # Initialize predictor at startup
    print("Initializing Name Matching Predictor...")
    try:
        get_predictor()
        print("Predictor initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        exit(1)

    # Run the Flask app
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,  # Set to False in production
    )

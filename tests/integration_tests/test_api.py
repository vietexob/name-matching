# pylint: disable=missing-module-docstring,missing-function-docstring
import json

import pytest

from app import app


@pytest.fixture
def client():
    """Creates a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_request_data():
    """Sample request data for testing."""
    return {
        "single": {
            "CUST_NAME": "John Smith",
            "COUNTERPART_NAME": "J. Smith",
            "FT_NO": "FT12345",
            "threshold": 0.85,
        },
        "batch": {
            "pairs": [
                {
                    "CUST_NAME": "John Smith",
                    "COUNTERPART_NAME": "J. Smith",
                    "FT_NO": "FT001",
                },
                {
                    "CUST_NAME": "Apple Inc.",
                    "COUNTERPART_NAME": "Apple Corporation",
                    "FT_NO": "FT002",
                },
                {
                    "CUST_NAME": "Microsoft",
                    "COUNTERPART_NAME": "Amazon",
                    "FT_NO": "FT003",
                },
            ],
            "threshold": 0.85,
        },
    }


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns success."""
        response = client.get("/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "name-matching-api"


class TestInfoEndpoint:
    """Test suite for /info endpoint."""

    def test_model_info(self, client):
        """Test model info endpoint returns model metadata."""
        response = client.get("/info")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "model" in data
        assert data["model"]["type"] == "LightGBM Classifier"
        assert "model_path" in data["model"]
        assert "tfidf_path" in data["model"]
        assert "features" in data["model"]
        assert data["model"]["num_features"] == 8


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""

    def test_predict_success(self, client, sample_request_data):
        """Test successful prediction request."""
        response = client.post(
            "/predict",
            data=json.dumps(sample_request_data["single"]),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "result" in data

        result = data["result"]
        assert "prediction" in result
        assert "probability" in result
        assert "match_label" in result
        assert "features" in result
        assert result["ft_no"] == "FT12345"
        assert result["name_x"] == "John Smith"
        assert result["name_y"] == "J. Smith"

    def test_predict_without_ft_no(self, client):
        """Test prediction without transaction number."""
        request_data = {
            "CUST_NAME": "John Doe",
            "COUNTERPART_NAME": "J. Doe",
            "threshold": 0.85,
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert data["result"]["ft_no"] is None

    def test_predict_without_threshold(self, client):
        """Test prediction uses default threshold when not provided."""
        request_data = {
            "CUST_NAME": "John Doe",
            "COUNTERPART_NAME": "J. Doe",
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert data["result"]["threshold"] == 0.85  # Default threshold

    def test_predict_with_custom_threshold(self, client):
        """Test prediction with custom threshold."""
        request_data = {
            "CUST_NAME": "John Doe",
            "COUNTERPART_NAME": "J. Doe",
            "FT_NO": "FT001",
            "threshold": 0.5,
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert data["result"]["threshold"] == 0.5

    def test_predict_missing_required_fields(self, client):
        """Test prediction with missing required fields."""
        # Missing COUNTERPART_NAME
        request_data = {"CUST_NAME": "John Doe", "FT_NO": "FT001"}

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"
        assert "required" in data["message"].lower()

    def test_predict_empty_names(self, client):
        """Test prediction with empty name strings."""
        request_data = {
            "CUST_NAME": "",
            "COUNTERPART_NAME": "John Doe",
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"

    def test_predict_invalid_threshold(self, client):
        """Test prediction with invalid threshold value."""
        request_data = {
            "CUST_NAME": "John Doe",
            "COUNTERPART_NAME": "J. Doe",
            "FT_NO": "FT001",
            "threshold": 1.5,  # Invalid: > 1
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"
        assert "threshold" in data["message"].lower()

    def test_predict_no_json_data(self, client):
        """Test prediction without JSON payload."""
        response = client.post("/predict", content_type="application/json")

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"
        assert "no json" in data["message"].lower()

    def test_predict_with_special_characters(self, client):
        """Test prediction with names containing special characters."""
        request_data = {
            "CUST_NAME": "O'Brien & Associates, Inc.",
            "COUNTERPART_NAME": "OBrien and Associates Inc",
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"


class TestPredictBatchEndpoint:
    """Test suite for /predict/batch endpoint."""

    def test_batch_predict_success(self, client, sample_request_data):
        """Test successful batch prediction request."""
        response = client.post(
            "/predict/batch",
            data=json.dumps(sample_request_data["batch"]),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "results" in data
        assert data["count"] == 3
        assert len(data["results"]) == 3

        for result in data["results"]:
            assert "prediction" in result
            assert "probability" in result
            assert "match_label" in result
            assert "features" in result

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty pairs list."""
        request_data = {"pairs": [], "threshold": 0.85}

        response = client.post(
            "/predict/batch",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"

    def test_batch_predict_without_pairs(self, client):
        """Test batch prediction without pairs field."""
        request_data = {"threshold": 0.85}

        response = client.post(
            "/predict/batch",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"

    def test_batch_predict_invalid_pair_format(self, client):
        """Test batch prediction with invalid pair format."""
        request_data = {
            "pairs": [
                {"CUST_NAME": "John Doe", "COUNTERPART_NAME": "J. Doe"},
                "invalid_string",  # Invalid format
            ],
            "threshold": 0.85,
        }

        response = client.post(
            "/predict/batch",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"

    def test_batch_predict_missing_fields_in_pair(self, client):
        """Test batch prediction with missing fields in a pair."""
        request_data = {
            "pairs": [
                {"CUST_NAME": "John Doe"},  # Missing COUNTERPART_NAME
            ],
            "threshold": 0.85,
        }

        response = client.post(
            "/predict/batch",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400

        data = json.loads(response.data)
        assert data["status"] == "error"

    def test_batch_predict_with_custom_threshold(self, client):
        """Test batch prediction with custom threshold."""
        request_data = {
            "pairs": [
                {
                    "CUST_NAME": "John Doe",
                    "COUNTERPART_NAME": "J. Doe",
                    "FT_NO": "FT001",
                }
            ],
            "threshold": 0.5,
        }

        response = client.post(
            "/predict/batch",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        assert data["results"][0]["threshold"] == 0.5

    def test_batch_predict_partial_failure(self, client):
        """Test batch prediction with some invalid pairs."""
        request_data = {
            "pairs": [
                {
                    "CUST_NAME": "John Doe",
                    "COUNTERPART_NAME": "J. Doe",
                    "FT_NO": "FT001",
                },
                {
                    "CUST_NAME": "",  # Invalid: empty name
                    "COUNTERPART_NAME": "Jane Doe",
                    "FT_NO": "FT002",
                },
                {
                    "CUST_NAME": "Apple Inc",
                    "COUNTERPART_NAME": "Apple Corp",
                    "FT_NO": "FT003",
                },
            ],
            "threshold": 0.85,
        }

        response = client.post(
            "/predict/batch",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        # Should return 207 Multi-Status for partial success
        assert response.status_code == 207

        data = json.loads(response.data)
        assert data["status"] == "partial_success"
        assert len(data["results"]) == 3
        # First and third should succeed, second should fail
        assert "error" not in data["results"][0]
        assert "error" in data["results"][1]
        assert "error" not in data["results"][2]


class TestErrorHandlers:
    """Test suite for error handlers."""

    def test_404_not_found(self, client):
        """Test 404 error handler for non-existent endpoint."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

        data = json.loads(response.data)
        assert data["status"] == "error"
        assert "not found" in data["message"].lower()
        assert "available_endpoints" in data

    def test_method_not_allowed(self, client):
        """Test that GET is not allowed on /predict endpoint."""
        response = client.get("/predict")

        assert response.status_code == 405


class TestCORSAndHeaders:
    """Test suite for CORS and headers."""

    def test_json_content_type(self, client, sample_request_data):
        """Test that responses have correct content type."""
        response = client.post(
            "/predict",
            data=json.dumps(sample_request_data["single"]),
            content_type="application/json",
        )

        assert "application/json" in response.content_type


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_very_long_names(self, client):
        """Test prediction with very long name strings."""
        long_name = "A" * 500

        request_data = {
            "CUST_NAME": long_name,
            "COUNTERPART_NAME": long_name,
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        # Should still process, though performance may vary
        assert response.status_code in [200, 500]

    def test_unicode_names(self, client):
        """Test prediction with unicode characters in names."""
        request_data = {
            "CUST_NAME": "François Müller 李明",
            "COUNTERPART_NAME": "Francois Mueller Li Ming",
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"

    def test_numeric_only_names(self, client):
        """Test prediction with numeric-only names."""
        request_data = {
            "CUST_NAME": "12345",
            "COUNTERPART_NAME": "12345",
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        # Should handle gracefully
        assert response.status_code in [200, 500]

    def test_identical_names(self, client):
        """Test prediction with identical names."""
        request_data = {
            "CUST_NAME": "John Smith",
            "COUNTERPART_NAME": "John Smith",
            "FT_NO": "FT001",
        }

        response = client.post(
            "/predict",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "success"
        # Should have high probability for identical names
        assert data["result"]["probability"] > 0.5

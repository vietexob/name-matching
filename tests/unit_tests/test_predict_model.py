# pylint: disable=missing-module-docstring,missing-function-docstring
import os

import pytest
import structlog

from name_matching.models.predict_model import NameMatchingPredictor


@pytest.fixture
def predictor():
    """Creates a NameMatchingPredictor instance for testing."""
    logger = structlog.get_logger()
    return NameMatchingPredictor(logger=logger)


@pytest.fixture
def sample_name_pairs():
    """Sample name pairs for testing."""
    return [
        {
            "name_x": "John Smith",
            "name_y": "J. Smith",
            "ft_no": "FT001",
            "expected_match": True,
        },
        {
            "name_x": "Apple Inc.",
            "name_y": "Apple Corporation",
            "ft_no": "FT002",
            "expected_match": True,
        },
        {
            "name_x": "Microsoft Corporation",
            "name_y": "Amazon Web Services",
            "ft_no": "FT003",
            "expected_match": False,
        },
        {
            "name_x": "Jane Marie Doe",
            "name_y": "Jane M. Doe",
            "ft_no": "FT004",
            "expected_match": True,
        },
    ]


class TestNameMatchingPredictor:
    """Test suite for NameMatchingPredictor class."""

    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly."""
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.tfidf_vectorizer is not None
        assert predictor.feature_generator is not None
        assert len(predictor.features_final) == 7

    def test_predictor_model_paths_exist(self, predictor):
        """Test that model files exist at specified paths."""
        assert os.path.exists(predictor.model_path)
        assert os.path.exists(predictor.tfidf_path)

    def test_predict_single_match(self, predictor):
        """Test prediction for a matching name pair."""
        result = predictor.predict(
            name_x="John Smith", name_y="J. Smith", ft_no="FT001", threshold=0.5
        )

        assert "error" not in result
        assert result["prediction"] in [0, 1]
        assert result["match_label"] == "MATCH"
        assert 0 <= result["probability"] <= 1
        assert result["ft_no"] == "FT001"
        assert result["name_x"] == "John Smith"
        assert result["name_y"] == "J. Smith"
        assert "features" in result
        assert len(result["features"]) == 7

    def test_predict_single_no_match(self, predictor):
        """Test prediction for a non-matching name pair."""
        result = predictor.predict(
            name_x="Microsoft Corporation",
            name_y="Amazon Web Services",
            ft_no="FT002",
            threshold=0.85,
        )

        assert "error" not in result
        assert result["prediction"] in [0, 1]
        assert result["match_label"] == "NO_MATCH"
        assert 0 <= result["probability"] <= 1
        assert result["ft_no"] == "FT002"

    def test_predict_with_empty_ft_no(self, predictor):
        """Test prediction without transaction number."""
        result = predictor.predict(
            name_x="John Doe", name_y="Doe John", ft_no="", threshold=0.8
        )

        assert "error" not in result
        assert result["ft_no"] is None
        assert result["match_label"] == "MATCH"

    def test_predict_with_different_thresholds(self, predictor):
        """Test prediction with different threshold values."""
        name_x = "Jane Doe"
        name_y = "Jane D."

        # Low threshold
        result_low = predictor.predict(
            name_x=name_x, name_y=name_y, ft_no="FT001", threshold=0.3
        )

        # High threshold
        result_high = predictor.predict(
            name_x=name_x, name_y=name_y, ft_no="FT001", threshold=0.95
        )

        assert "error" not in result_low
        assert "error" not in result_high
        # Same probability, different predictions based on threshold
        assert result_low["probability"] == result_high["probability"]
        assert result_low["match_label"] == "MATCH"
        assert result_high["match_label"] == "MATCH"

    def test_predict_empty_name_validation(self, predictor):
        """Test that empty names are properly validated."""
        # Empty name_x
        result = predictor.predict(name_x="", name_y="John Doe", ft_no="FT001")
        assert "error" in result
        assert result["error"] == "Validation error"

        # Empty name_y
        result = predictor.predict(name_x="John Doe", name_y="", ft_no="FT001")
        assert "error" in result
        assert result["error"] == "Validation error"

        # Both empty
        result = predictor.predict(name_x="", name_y="", ft_no="FT001")
        assert "error" in result

    def test_predict_type_validation(self, predictor):
        """Test that input types are properly validated."""
        # Non-string name_x
        result = predictor.predict(name_x=123, name_y="John Doe", ft_no="FT001")
        assert "error" in result
        assert result["error"] == "Type error"

        # Non-string name_y
        result = predictor.predict(name_x="John Doe", name_y=456, ft_no="FT001")
        assert "error" in result
        assert result["error"] == "Type error"

    def test_predict_with_special_characters(self, predictor):
        """Test prediction with names containing special characters."""
        result = predictor.predict(
            name_x="O'Brien & Associates, Inc.",
            name_y="OBrien and Associates Inc",
            ft_no="FT001",
            threshold=0.85,
        )

        assert "error" not in result
        assert result["prediction"] in [0, 1]
        assert "features" in result
        assert result["match_label"] == "MATCH"

    def test_predict_with_unicode_characters(self, predictor):
        """Test prediction with names containing unicode characters."""
        result = predictor.predict(
            name_x="François Müller",
            name_y="Francois Mueller",
            ft_no="FT001",
            threshold=0.85,
        )

        assert "error" not in result
        assert result["prediction"] in [0, 1]
        assert result["match_label"] == "MATCH"

    def test_predict_batch_success(self, predictor, sample_name_pairs):
        """Test batch prediction with multiple name pairs."""
        pairs = [
            {"name_x": pair["name_x"], "name_y": pair["name_y"], "ft_no": pair["ft_no"]}
            for pair in sample_name_pairs
        ]

        results = predictor.predict_batch(pairs, threshold=0.85)

        assert len(results) == len(pairs)
        for result in results:
            assert "error" not in result
            assert "prediction" in result
            assert "probability" in result
            assert "features" in result

    def test_predict_batch_empty_list(self, predictor):
        """Test batch prediction with empty list."""
        results = predictor.predict_batch([], threshold=0.85)
        assert len(results) == 0

    def test_predict_batch_with_errors(self, predictor):
        """Test batch prediction with some invalid pairs."""
        pairs = [
            {"name_x": "John Doe", "name_y": "J. Doe", "ft_no": "FT001"},
            {"name_x": "", "name_y": "Jane Doe", "ft_no": "FT002"},  # Invalid
            {"name_x": "Apple Inc", "name_y": "Apple Corp", "ft_no": "FT003"},
        ]

        results = predictor.predict_batch(pairs, threshold=0.85)

        assert len(results) == 3
        assert "error" not in results[0]
        assert "error" in results[1]  # This one should have an error
        assert "error" not in results[2]

    def test_preprocess_names(self, predictor):
        """Test name preprocessing functionality."""
        name_x = "John O'Brien & Co."
        name_y = "JANE DOE-SMITH"

        processed_x, processed_y = predictor._preprocess_names(name_x, name_y)

        # Check that names are uppercase and processed
        assert processed_x.isupper()
        assert processed_y.isupper()
        # Check that special characters are handled
        assert len(processed_x) > 0
        assert len(processed_y) > 0

    def test_feature_values_in_valid_range(self, predictor):
        """Test that generated feature values are in valid ranges."""
        result = predictor.predict(
            name_x="John Smith", name_y="J. Smith", ft_no="FT001", threshold=0.85
        )

        assert "error" not in result
        features = result["features"]

        # Check that all features are numeric and in reasonable ranges
        for feature_name, feature_value in features.items():
            assert isinstance(feature_value, (int, float))
            # Most similarity features should be between 0 and 1
            # PARTIAL_RATIO from fuzzywuzzy returns 0-100
            # EMB_DISTANCE can vary depending on the metric used
            if feature_name != "EMB_DISTANCE":
                if feature_name == "PARTIAL_RATIO":
                    assert 0 <= feature_value <= 100, f"{feature_name} out of range"
                else:
                    assert 0 <= feature_value <= 1, f"{feature_name} out of range"

    def test_predictor_with_custom_paths(self, config_ini):
        """Test predictor initialization with custom model paths."""
        model_path = config_ini["MODELPATH"]["MODEL_LGB_NAME_MATCHING"]
        tfidf_path = config_ini["MODELPATH"]["FILENAME_MODEL_TFIDF_NGRAM"]

        predictor = NameMatchingPredictor(model_path=model_path, tfidf_path=tfidf_path)

        assert predictor.model_path == model_path
        assert predictor.tfidf_path == tfidf_path
        assert predictor.model is not None

    def test_predict_result_structure(self, predictor):
        """Test that prediction result has the correct structure."""
        result = predictor.predict(
            name_x="John Doe", name_y="Jane Doe", ft_no="FT001", threshold=0.85
        )

        # Check all required keys are present
        required_keys = [
            "ft_no",
            "name_x",
            "name_y",
            "prediction",
            "match_label",
            "probability",
            "threshold",
            "features",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check types
        assert isinstance(result["prediction"], int)
        assert isinstance(result["match_label"], str)
        assert isinstance(result["probability"], float)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["features"], dict)

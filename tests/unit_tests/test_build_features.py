# pylint: disable=missing-module-docstring,missing-function-docstring
import pickle

import editdistance
import numpy as np
import pytest
import structlog

from name_matching.features.build_features import (
    FeatureGenerator,
    compute_cosine_sims,
    compute_embedding_distances,
    compute_jaccard_sim,
    get_ratio_feature,
)


@pytest.fixture
def setup(config_ini):
    names_x = ["John Doe", "Richard Smith", "Ben Owens", ""]
    names_y = ["Jane Doe", "Richard Adams", "Sophia Hills", "Harry Porter"]
    filename_tfidf_model = config_ini["MODELPATH"]["FILENAME_MODEL_TFIDF_NGRAM"]
    with open(filename_tfidf_model, "rb") as model_file:
        tfidf_vect_ngram = pickle.load(model_file)
    return names_x, names_y, tfidf_vect_ngram


@pytest.fixture
def feature_generator():
    logger = structlog.get_logger()
    return FeatureGenerator(logger=logger)


def test_get_ratio_feature(setup):
    names_x, names_y, _ = setup
    edit_dist_xy = [
        editdistance.eval(x.strip(), y.strip()) for x, y in zip(names_x, names_y)
    ]
    ratio_features = get_ratio_feature(names_x, names_y, edit_dist_xy)
    lower_bound_checks = [ratio >= 0 for ratio in ratio_features]
    upper_bound_checks = [ratio <= 1 for ratio in ratio_features]

    assert sum(lower_bound_checks) == len(names_x)
    assert sum(upper_bound_checks) == len(names_x)


def test_compute_jaccard_sim():
    name_x = "Lawrence Arabia"
    name_y = "Yves Saint Laurent"

    assert 0 <= compute_jaccard_sim(name_x, name_y) <= 1
    assert compute_jaccard_sim("", name_y) == 0


def test_compute_jaccard_sim_both_empty():
    """Test Jaccard similarity with both empty strings"""
    assert compute_jaccard_sim("", "") == 0


def test_compute_jaccard_sim_no_common_tokens():
    """Test Jaccard similarity with no common tokens"""
    name_x = "John Doe"
    name_y = "Alice Smith"
    result = compute_jaccard_sim(name_x, name_y)
    assert result == 0


def test_compute_jaccard_sim_identical():
    """Test Jaccard similarity with identical names"""
    name = "John Doe"
    result = compute_jaccard_sim(name, name)
    assert result == 1.0


def test_compute_cosine_sims(setup):
    names_x, names_y, tfidf_vect_ngram = setup
    cosine_sims = compute_cosine_sims(names_x, names_y, tfidf_vect_ngram)
    lower_bound_checks = [ratio >= 0 for ratio in cosine_sims]
    upper_bound_checks = [ratio <= 1 for ratio in cosine_sims]

    assert sum(lower_bound_checks) == len(names_x)
    assert sum(upper_bound_checks) == len(names_x)


def test_compute_cosine_sims_assertion_error(setup):
    """Test that cosine_sims raises assertion error on mismatched lengths"""
    _, _, tfidf_vect_ngram = setup
    names_x = ["John Doe"]
    names_y = ["Jane Doe", "Alice Smith"]

    with pytest.raises(AssertionError):
        compute_cosine_sims(names_x, names_y, tfidf_vect_ngram)


def test_compute_embedding_distances_cosine():
    """Test embedding distances with cosine similarity"""
    names_x = ["John Doe", "Alice Smith"]
    names_y = ["Jane Doe", "Bob Jones"]

    distances = compute_embedding_distances(names_x, names_y, cosine_dist=True)

    assert len(distances) == len(names_x)
    # The function returns numpy float values, check that they're numeric

    assert all(isinstance(d, (float, int, np.floating, np.integer)) for d in distances)


def test_compute_embedding_distances_euclidean():
    """Test embedding distances with Euclidean distance"""
    names_x = ["John Doe", "Alice Smith"]
    names_y = ["Jane Doe", "Bob Jones"]

    distances = compute_embedding_distances(names_x, names_y, cosine_dist=False)

    assert len(distances) == len(names_x)

    # The function returns numpy float values, check that they're numeric
    assert all(isinstance(d, (float, int, np.floating, np.integer)) for d in distances)


def test_compute_embedding_distances_assertion_error():
    """Test that embedding distances raises assertion error on mismatched lengths"""
    names_x = ["John Doe"]
    names_y = ["Jane Doe", "Alice Smith"]

    with pytest.raises(AssertionError):
        compute_embedding_distances(names_x, names_y)


def test_feature_generator_init(feature_generator):
    """Test FeatureGenerator initialization"""
    assert feature_generator is not None
    assert hasattr(feature_generator, "logger")
    assert hasattr(feature_generator, "jaccard_sim_col")


def test_feature_generator_init_with_none_logger():
    """Test FeatureGenerator initialization with None logger"""
    fg = FeatureGenerator(logger=None)
    assert fg.logger is not None


def test_feature_generator_build_features(feature_generator, setup):
    """Test FeatureGenerator.build_features with valid inputs"""
    names_x = ["John Doe", "Richard Smith"]
    names_y = ["Jane Doe", "Richard Adams"]
    _, _, tfidf_vect_ngram = setup

    df_featured = feature_generator.build_features(names_x, names_y, tfidf_vect_ngram)

    assert df_featured is not None
    assert len(df_featured) == len(names_x)
    assert feature_generator.jaccard_sim_col in df_featured.columns
    assert feature_generator.cosine_sim_col in df_featured.columns
    assert feature_generator.ratio_col in df_featured.columns
    assert feature_generator.sorted_token_ratio_col in df_featured.columns
    assert feature_generator.token_set_ratio_col in df_featured.columns
    assert feature_generator.partial_ratio_col in df_featured.columns
    assert feature_generator.emb_dist_col in df_featured.columns


def test_feature_generator_build_features_mismatched_lengths(feature_generator, setup):
    """Test build_features with mismatched input lengths"""
    names_x = ["John Doe"]
    names_y = ["Jane Doe", "Alice Smith"]
    _, _, tfidf_vect_ngram = setup

    df_featured = feature_generator.build_features(names_x, names_y, tfidf_vect_ngram)

    assert df_featured is None


def test_feature_generator_build_features_empty_list(feature_generator, setup):
    """Test build_features with empty list"""
    names_x = []
    names_y = []
    _, _, tfidf_vect_ngram = setup

    df_featured = feature_generator.build_features(names_x, names_y, tfidf_vect_ngram)

    assert df_featured is None


def test_feature_generator_build_features_empty_strings(feature_generator, setup):
    """Test build_features with empty strings in names"""
    names_x = ["John Doe", ""]
    names_y = ["Jane Doe", "Alice Smith"]
    _, _, tfidf_vect_ngram = setup

    df_featured = feature_generator.build_features(names_x, names_y, tfidf_vect_ngram)

    assert df_featured is None


def test_feature_generator_create_tfidf_vectorizer(feature_generator, tmp_path):
    """Test create_tfidf_vectorizer with valid corpus"""
    all_names = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown"]

    # Mock the config to use a temporary path
    import name_matching.features.build_features as bf

    original_config = bf.config
    bf.config = {
        "MODELPATH": {"FILENAME_MODEL_TFIDF_NGRAM": str(tmp_path / "test_tfidf.pkl")}
    }

    try:
        vectorizer = feature_generator.create_tfidf_vectorizer(all_names)

        assert vectorizer is not None
        assert hasattr(vectorizer, "transform")
        assert (tmp_path / "test_tfidf.pkl").exists()
    finally:
        # Restore original config
        bf.config = original_config


def test_feature_generator_create_tfidf_vectorizer_empty_corpus(feature_generator):
    """Test create_tfidf_vectorizer with empty corpus"""
    all_names = []

    vectorizer = feature_generator.create_tfidf_vectorizer(all_names)

    assert vectorizer is None

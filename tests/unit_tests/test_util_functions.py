# pylint: disable=missing-module-docstring,missing-function-docstring

import pytest

from name_matching.config import read_config

config = read_config()

from name_matching.utils.utils import (
    generate_typo_name,
    plot_precision_recall_auc,
    plot_roc_auc,
    process_text_standard,
    remove_or_extract_numeric_tokens,
)


def test_process_text_standard():
    text = "THE CLIENT ### WAS BORN ON 02/01 YEAR 1989."
    processed = "CLIENT BORN YEAR"
    assert (
        process_text_standard(
            text, remove_numbers=True, remove_stopwords=True, stem=False
        )
        == processed
    )
    processed = "CLIENT BORN 02 01 YEAR 1989"
    assert (
        process_text_standard(
            text, remove_numbers=False, remove_stopwords=True, stem=False
        )
        == processed
    )
    processed = "THE CLIENT WAS BORN ON YEAR"
    assert (
        process_text_standard(
            text, remove_numbers=True, remove_stopwords=False, stem=False
        )
        == processed
    )
    processed = "THE CLIENT WAS BORN ON 02 01 YEAR 1989"
    assert (
        process_text_standard(
            text, remove_numbers=False, remove_stopwords=False, stem=False
        )
        == processed
    )


def test_process_text_standard_with_stem():
    """Test text processing with stemming enabled"""
    text = "RUNNING QUICKLY THROUGH TREES"
    result = process_text_standard(
        text, remove_numbers=True, remove_stopwords=True, stem=True
    )
    # Stemming should reduce words to their root form
    assert isinstance(result, str)
    assert len(result) > 0


def test_remove_or_extract_numeric_tokens():
    text = "Token 1234 removed"
    assert remove_or_extract_numeric_tokens(text, is_removal=True) == "Token removed"
    assert remove_or_extract_numeric_tokens(text, is_removal=False) == "1234"
    text = "Token 12-34 removed"
    assert (
        # Because "12-34" is alphanumeric, but strictly numeric
        remove_or_extract_numeric_tokens(text, is_removal=True)
        == "Token 12-34 removed"
    )


def test_remove_or_extract_numeric_tokens_multiple():
    """Test extracting multiple numeric tokens"""
    text = "Token 123 and 456 and 789"
    extracted = remove_or_extract_numeric_tokens(text, is_removal=False)
    assert "123" in extracted
    assert "456" in extracted
    assert "789" in extracted


def test_remove_or_extract_numeric_tokens_no_numbers():
    """Test with text containing no numbers"""
    text = "No numbers here"
    assert remove_or_extract_numeric_tokens(text, is_removal=True) == text
    assert remove_or_extract_numeric_tokens(text, is_removal=False) == ""


def test_generate_typo_name():
    """Test typo generation function"""
    name = "JOHN DOE"
    typo = generate_typo_name(name, prob_flip=0.5)

    # Result should be a string
    assert isinstance(typo, str)
    # Result should have similar length (may vary slightly)
    assert len(typo) > 0


def test_generate_typo_name_empty_string():
    """Test typo generation with empty string"""
    name = ""
    typo = generate_typo_name(name, prob_flip=0.5)
    assert typo == ""


def test_generate_typo_name_invalid_probability():
    """Test typo generation with invalid probability"""
    name = "JOHN DOE"

    with pytest.raises(AssertionError):
        generate_typo_name(name, prob_flip=0.0)

    with pytest.raises(AssertionError):
        generate_typo_name(name, prob_flip=1.0)

    with pytest.raises(AssertionError):
        generate_typo_name(name, prob_flip=1.5)


def test_generate_typo_name_low_probability():
    """Test typo generation with very low probability"""
    name = "JOHN DOE"
    # With very low probability, name might remain unchanged
    typo = generate_typo_name(name, prob_flip=0.01)
    assert isinstance(typo, str)
    # Should still have reasonable length
    assert len(typo) >= len(name) - 1


def test_plot_roc_auc(tmp_path):
    """Test ROC-AUC plotting function"""
    y_test = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    y_pred_prob = [0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.85, 0.15, 0.75, 0.95]

    # Test without saving
    plot_roc_auc(y_test, y_pred_prob, filename_out="")

    # Test with saving to file
    output_file = tmp_path / "roc_curve.png"
    plot_roc_auc(y_test, y_pred_prob, filename_out=str(output_file))
    assert output_file.exists()


def test_plot_roc_auc_mismatched_lengths():
    """Test ROC-AUC plotting with mismatched input lengths"""
    y_test = [0, 1, 1]
    y_pred_prob = [0.1, 0.8]

    with pytest.raises(AssertionError):
        plot_roc_auc(y_test, y_pred_prob)


def test_plot_precision_recall_auc(tmp_path):
    """Test Precision-Recall AUC plotting function"""
    y_test = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    y_pred_prob = [0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.85, 0.15, 0.75, 0.95]

    # Test without saving
    pr_auc = plot_precision_recall_auc(y_test, y_pred_prob, filename_out="")
    assert isinstance(pr_auc, float)
    assert 0 <= pr_auc <= 100

    # Test with saving to file
    output_file = tmp_path / "pr_curve.png"
    pr_auc = plot_precision_recall_auc(
        y_test, y_pred_prob, filename_out=str(output_file)
    )
    assert output_file.exists()
    assert isinstance(pr_auc, float)


def test_plot_precision_recall_auc_mismatched_lengths():
    """Test Precision-Recall plotting with mismatched input lengths"""
    y_test = [0, 1, 1]
    y_pred_prob = [0.1, 0.8]

    with pytest.raises(AssertionError):
        plot_precision_recall_auc(y_test, y_pred_prob)

import os
import pickle
import warnings
from typing import Any, Dict, Optional, Tuple

import structlog
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from name_matching.config import read_config
from name_matching.features.build_features import FeatureGenerator
from name_matching.log.logging import configure_structlog
from name_matching.utils.utils import process_text_standard

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = read_config()

# Suppress warnings
warnings.filterwarnings("ignore")


class NameMatchingPredictor:
    """
    Class for real-time name matching prediction/inference.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        model_path: Optional[str] = None,
        tfidf_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the predictor by loading the trained model and TF-IDF vectorizer.

        :param logger: The logging object (optional)
        :param model_path: Path to the trained LightGBM model pickle file (optional)
        :param tfidf_path: Path to the TF-IDF vectorizer pickle file (optional)
        """

        self.logger = logger if logger is not None else structlog.get_logger()

        # Set model paths from config if not provided
        self.model_path = (
            model_path
            if model_path is not None
            else config["MODELPATH"]["MODEL_LGB_NAME_MATCHING"]
        )
        self.tfidf_path = (
            tfidf_path
            if tfidf_path is not None
            else config["MODELPATH"]["FILENAME_MODEL_TFIDF_NGRAM"]
        )

        # Feature columns (same order as training)
        self.features_final = [
            config["DATA.COLUMNS"]["JACCARD_SIM_COL"],
            config["DATA.COLUMNS"]["COSINE_SIM_COL"],
            config["DATA.COLUMNS"]["RATIO_COL"],
            config["DATA.COLUMNS"]["SORTED_TOKEN_RATIO_COL"],
            config["DATA.COLUMNS"]["TOKEN_SET_RATIO_COL"],
            config["DATA.COLUMNS"]["PARTIAL_RATIO_COL"],
            config["DATA.COLUMNS"]["EMB_DISTANCE_COL"],
        ]

        # Load the trained model and vectorizer
        self.model: Optional[LGBMClassifier] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._load_models()

        # Initialize feature generator
        self.feature_generator = FeatureGenerator(self.logger)

    def _load_models(self) -> None:
        """
        Loads the trained LightGBM model and TF-IDF vectorizer from disk.

        :raises FileNotFoundError: If model files are not found
        :raises Exception: If there's an error loading the models
        """

        try:
            # Load the LightGBM model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at: {self.model_path}. "
                    f"Please train the model first using train_model.py"
                )

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            self.logger.info("LOADED_MODEL", path=self.model_path)

            # Load the TF-IDF vectorizer
            if not os.path.exists(self.tfidf_path):
                raise FileNotFoundError(
                    f"TF-IDF vectorizer file not found at: {self.tfidf_path}. "
                    f"Please train the model first using train_model.py"
                )

            with open(self.tfidf_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            self.logger.info("LOADED_TFIDF_VECTORIZER", path=self.tfidf_path)

        except Exception as e:
            self.logger.error("MODEL_LOADING_ERROR", error=str(e))
            raise

    def _preprocess_names(self, name_x: str, name_y: str) -> Tuple[str, str]:
        """
        Preprocesses the input names using standard text normalization.

        :param name_x: First name (customer name)
        :param name_y: Second name (counterpart name)
        :return: Tuple of preprocessed names
        """

        # Convert to uppercase
        name_x = name_x.upper()
        name_y = name_y.upper()

        # Apply standard text preprocessing (same as training)
        name_x_processed = process_text_standard(name_x, remove_stopwords=False)
        name_y_processed = process_text_standard(name_y, remove_stopwords=False)

        return name_x_processed, name_y_processed

    def predict(
        self, name_x: str, name_y: str, ft_no: str = "", threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Predicts whether two names refer to the same entity.

        :param name_x: First name (customer name)
        :param name_y: Second name (counterpart name)
        :param ft_no: Transaction reference number (optional, for logging/tracking)
        :param threshold: Classification threshold for positive class (default: 0.85)
        :return: Dictionary containing prediction results
        """

        try:
            # Validate inputs
            if not name_x or not name_y:
                raise ValueError("Both name_x and name_y must be non-empty strings")

            if not isinstance(name_x, str) or not isinstance(name_y, str):
                raise TypeError("Both name_x and name_y must be strings")

            # Log the prediction request
            self.logger.info(
                "PREDICTION_REQUEST",
                name_x=name_x,
                name_y=name_y,
                ft_no=ft_no if ft_no else "N/A",
            )

            # Preprocess the names
            name_x_processed, name_y_processed = self._preprocess_names(name_x, name_y)

            # Generate features
            df_features = self.feature_generator.build_features(
                [name_x_processed], [name_y_processed], self.tfidf_vectorizer
            )

            if df_features is None or df_features.empty:
                raise Exception("Feature generation failed")

            # Extract feature values in the correct order
            features = df_features[self.features_final].iloc[0].values.reshape(1, -1)

            # Get prediction probability
            prob = self.model.predict_proba(features)[0, 1]

            # Apply threshold to get binary prediction
            prediction = 1 if prob >= threshold else 0
            match_label = "MATCH" if prediction == 1 else "NO_MATCH"

            # Prepare response
            result = {
                "ft_no": ft_no if ft_no else None,
                "name_x": name_x,
                "name_y": name_y,
                "prediction": prediction,
                "match_label": match_label,
                "probability": round(float(prob), 4),
                "threshold": threshold,
                "features": {
                    feature: round(float(df_features[feature].iloc[0]), 4)
                    for feature in self.features_final
                },
            }

            self.logger.info(
                "PREDICTION_RESULT",
                ft_no=ft_no if ft_no else "N/A",
                prediction=match_label,
                probability=round(float(prob), 4),
            )

            return result

        except ValueError as e:
            self.logger.error("VALIDATION_ERROR", error=str(e))
            return {
                "error": "Validation error",
                "message": str(e),
                "ft_no": ft_no if ft_no else None,
            }
        except TypeError as e:
            self.logger.error("TYPE_ERROR", error=str(e))
            return {
                "error": "Type error",
                "message": str(e),
                "ft_no": ft_no if ft_no else None,
            }
        except Exception as e:
            self.logger.error("PREDICTION_ERROR", error=str(e), ft_no=ft_no)
            return {
                "error": "Prediction error",
                "message": str(e),
                "ft_no": ft_no if ft_no else None,
            }

    def predict_batch(
        self, name_pairs: list, threshold: float = 0.85
    ) -> list[Dict[str, Any]]:
        """
        Predicts matches for multiple name pairs.

        :param name_pairs: List of dictionaries with keys 'name_x', 'name_y', and optionally 'ft_no'
        :param threshold: Classification threshold for positive class (default: 0.85)
        :return: List of prediction result dictionaries
        """

        results: list[Dict[str, Any]] = [None] * len(name_pairs)
        valid_pairs = []
        names_x_processed = []
        names_y_processed = []

        # Validate and preprocess all name pairs
        for idx, pair in enumerate(name_pairs):
            name_x = pair.get("name_x", "")
            name_y = pair.get("name_y", "")
            ft_no = pair.get("ft_no", "")

            try:
                if not name_x or not name_y:
                    raise ValueError("Both name_x and name_y must be non-empty strings")
                if not isinstance(name_x, str) or not isinstance(name_y, str):
                    raise TypeError("Both name_x and name_y must be strings")

                self.logger.info(
                    "PREDICTION_REQUEST",
                    name_x=name_x,
                    name_y=name_y,
                    ft_no=ft_no if ft_no else "N/A",
                )

                proc_name_x, proc_name_y = self._preprocess_names(name_x, name_y)
                names_x_processed.append(proc_name_x)
                names_y_processed.append(proc_name_y)
                valid_pairs.append((idx, name_x, name_y, ft_no))
            except ValueError as e:
                self.logger.error("VALIDATION_ERROR", error=str(e))
                results[idx] = {
                    "error": "Validation error",
                    "message": str(e),
                    "ft_no": ft_no if ft_no else None,
                }
            except TypeError as e:
                self.logger.error("TYPE_ERROR", error=str(e))
                results[idx] = {
                    "error": "Type error",
                    "message": str(e),
                    "ft_no": ft_no if ft_no else None,
                }
            except Exception as e:
                self.logger.error("PREDICTION_ERROR", error=str(e), ft_no=ft_no)
                results[idx] = {
                    "error": "Prediction error",
                    "message": str(e),
                    "ft_no": ft_no if ft_no else None,
                }

        if not valid_pairs:
            return results

        try:
            # Generate features for all valid pairs
            df_features = self.feature_generator.build_features(
                names_x_processed, names_y_processed, self.tfidf_vectorizer
            )

            if df_features is None or df_features.empty:
                raise Exception("Feature generation failed")
            
            # Get prediction probabilities
            feature_matrix = df_features[self.features_final].values
            probs = self.model.predict_proba(feature_matrix)[:, 1]

            # Compile results
            for i, (idx, name_x, name_y, ft_no) in enumerate(valid_pairs):
                prob = float(probs[i])
                prediction = 1 if prob >= threshold else 0
                match_label = "MATCH" if prediction == 1 else "NO_MATCH"

                results[idx] = {
                    "ft_no": ft_no if ft_no else None,
                    "name_x": name_x,
                    "name_y": name_y,
                    "prediction": prediction,
                    "match_label": match_label,
                    "probability": round(prob, 4),
                    "threshold": threshold,
                    "features": {
                        feature: round(float(df_features.iloc[i][feature]), 4)
                        for feature in self.features_final
                    },
                }

                self.logger.info(
                    "PREDICTION_RESULT",
                    ft_no=ft_no if ft_no else "N/A",
                    prediction=match_label,
                    probability=round(prob, 4),
                )
        except Exception as e:
            for idx, _, _, ft_no in valid_pairs:
                results[idx] = {
                    "error": "Prediction error",
                    "message": str(e),
                    "ft_no": ft_no if ft_no else None,
                }
            self.logger.error("PREDICTION_ERROR", error=str(e))

        return results


def main():
    """Example usage of the NameMatchingPredictor class."""

    # Configure logging
    configure_structlog(silent=False)
    logger = structlog.get_logger()

    # Create predictor instance
    predictor = NameMatchingPredictor(logger=logger)

    # Example 1: Single prediction
    logger.info("EXAMPLE_1_SINGLE_PREDICTION")
    result = predictor.predict(
        name_x="John Smith", name_y="J. Smith", ft_no="FT12345", threshold=0.85
    )
    print(f"\nResult: {result}")

    # Example 2: Batch prediction
    logger.info("EXAMPLE_2_BATCH_PREDICTION")
    name_pairs = [
        {"name_x": "Apple Inc.", "name_y": "Apple Corporation", "ft_no": "FT001"},
        {"name_x": "Microsoft", "name_y": "Amazon", "ft_no": "FT002"},
        {"name_x": "Jane Doe", "name_y": "Jane D.", "ft_no": "FT003"},
    ]
    results = predictor.predict_batch(name_pairs, threshold=0.85)
    print(f"\nBatch Results:")
    for result in results:
        print(f"  {result}")


if __name__ == "__main__":
    main()

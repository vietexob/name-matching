import pickle
from typing import Any, List

import editdistance
import pandas as pd
import structlog
from fuzzywuzzy import fuzz  # for partial (token) ratio metric
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ML libraries
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from name_matching.config import read_config

# Instantiate configuration class
config = read_config()


def get_ratio_feature(
    names_x: List[str], names_y: List[str], edit_dist_xy: List[int]
) -> List[float]:
    """
    Computes the ratio features between two sets of string vectors.

    :param names_x: List of left-side string names
    :param names_y: List of right-side string names
    :param edit_dist_xy: List of edit distances between left-side names and right-side names
    :return: List of ratio features (values between 0 and 1)
    """

    # Adding 1 in order to avoid division by zero error
    ratio_denom = [
        max(len(name_x), len(name_y), 1) for name_x, name_y in zip(names_x, names_y)
    ]

    ratio_feature = [
        1 - (float(delta) / denom) for delta, denom in zip(edit_dist_xy, ratio_denom)
    ]

    return ratio_feature


def compute_jaccard_sim(name_x: str, name_y: str) -> float:
    """
    Computes the Jaccard similarity (IoU) between two given strings.

    :param name_x: Left-side string
    :param name_y: Right-side string
    :return: Jaccard similarity (value between 0 and 1)
    """

    if len(name_x) == 0 or len(name_y) == 0:
        return 0

    tokens_x = name_x.split()
    tokens_y = name_y.split()
    commons = set(tokens_x).intersection(set(tokens_y))

    if len(commons) > 0:
        return float(len(commons)) / (len(tokens_x) + len(tokens_y) - len(commons))

    return 0


def compute_cosine_sims(
    names_x: List[str], names_y: List[str], tfidf_vectorizer: TfidfVectorizer
) -> List[float]:
    """
    Computes the cosine similarities between two sets of string vectors.

    :param names_x: List of left-side string names
    :param names_y: List of right-side string names
    :param tfidf_vectorizer: TF-IDF vectorizer object
    :return: List of cosine similarities (values between 0 and 1)
    """

    assert len(names_x) == len(names_y), "Input lists are not of the same length!"

    tfidf_names_x = tfidf_vectorizer.transform(names_x)
    tfidf_names_y = tfidf_vectorizer.transform(names_y)
    cosine_sims = [
        cosine_similarity(x, y)[0, 0] for x, y in zip(tfidf_names_x, tfidf_names_y)
    ]

    return cosine_sims


def compute_embedding_distances(
    names_x: List[str], names_y: List[str], cosine_dist: bool = True
) -> List[float]:
    """
    Computes the embedding distances between two sets of string vectors.

    Args:
        names_x (List[str]): _description_
        names_y (List[str]): _description_
        cosine_dist (bool): _description_

    Returns:
        List[float]: _description_
    """

    assert len(names_x) == len(names_y), "Input lists are not of the same length!"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_names_x = model.encode(names_x)
    emb_names_y = model.encode(names_y)

    if cosine_dist:
        emb_dist = cosine_similarity(emb_names_x, emb_names_y)
    else:
        emb_dist = euclidean_distances(emb_names_x, emb_names_y)

    res = [emb_dist[i, i] for i in range(len(names_x))]
    return res


class FeatureGenerator:
    """
    Class for feature generation (feature engineering) at training and inference
    and for creating TF-IDF vectorizer object on string names.
    """

    def __init__(self, logger: Any) -> None:
        """
        Inits FeatureGenerator class.

        :param logger: The logging object
        """

        # Set the instance variables
        self.logger = logger if logger is not None else structlog.get_logger()

        # Feature columns
        self.jaccard_sim_col = config["DATA.COLUMNS"]["JACCARD_SIM_COL"]
        self.cosine_sim_col = config["DATA.COLUMNS"]["COSINE_SIM_COL"]
        self.ratio_col = config["DATA.COLUMNS"]["RATIO_COL"]
        self.sorted_token_ratio_col = config["DATA.COLUMNS"]["SORTED_TOKEN_RATIO_COL"]
        self.token_set_ratio_col = config["DATA.COLUMNS"]["TOKEN_SET_RATIO_COL"]
        self.partial_ratio_col = config["DATA.COLUMNS"]["PARTIAL_RATIO_COL"]
        self.emb_dist_col = config["DATA.COLUMNS"]["EMB_DISTANCE_COL"]
        self.len_diff_col = config["DATA.COLUMNS"]["LEN_DIFF_COL"]

    def build_features(
        self, names_x: List[str], names_y: List[str], tfidf_vectorizer: TfidfVectorizer
    ) -> pd.DataFrame:
        """
        Creates a featured data frame to be used for training and inference.

        :param names_x: List of left-side string names
        :param names_y: List of right-side string names
        :param tfidf_vectorizer: TF-IDF vectorizer object
        :return: Featured data frame
        """

        try:
            # Make sure there are no empty lists
            assert len(names_x) == len(
                names_y
            ), "Input lists are not of the same length!"
            assert len(names_x) > 0, "Input list is empty!"

            # Make sure there are no empty strings
            assert all(len(name) > 0 for name in names_x), "Empty name string detected!"
            assert all(len(name) > 0 for name in names_y), "Empty name string detected!"

            # TF-IDF features
            self.logger.info("GENERATING_TF_IDF_FEATURES")
            cosine_sims_xy = compute_cosine_sims(names_x, names_y, tfidf_vectorizer)

            # Jaccard similarity -- intersection over union of tokens
            self.logger.info("GENERATING_JACCARD_SIMILIARITY_FEATURES")
            jaccard_xy = [
                compute_jaccard_sim(x.strip(), y.strip())
                for x, y in zip(names_x, names_y)
            ]

            # Ratio feature
            self.logger.info("GENERATING_EDIT_DISTANCE_FEATURES")
            edit_dist_xy = [
                editdistance.eval(x.strip(), y.strip())
                for x, y in zip(names_x, names_y)
            ]
            ratio_features = get_ratio_feature(names_x, names_y, edit_dist_xy)

            # Sorted token ratio
            sorted_names_x = [" ".join(sorted(name.split())) for name in names_x]
            sorted_names_y = [" ".join(sorted(name.split())) for name in names_y]
            sorted_edit_dist_xy = [
                editdistance.eval(x.strip(), y.strip())
                for x, y in zip(sorted_names_x, sorted_names_y)
            ]
            sorted_token_ratios = get_ratio_feature(
                sorted_names_x, sorted_names_y, sorted_edit_dist_xy
            )

            # Token set ratio
            set_names_x = [" ".join(sorted(set(name.split()))) for name in names_x]
            set_names_y = [" ".join(sorted(set(name.split()))) for name in names_y]
            set_edit_dist_xy = [
                editdistance.eval(x.strip(), y.strip())
                for x, y in zip(set_names_x, set_names_y)
            ]
            token_set_ratios = get_ratio_feature(
                set_names_x, set_names_y, set_edit_dist_xy
            )

            # Partial (token) ratio
            partial_ratios = [
                fuzz.partial_ratio(names_x, names_y)
                for names_x, names_y in zip(names_x, names_y)
            ]

            # Embedding distance (cosine or Euclidean)
            self.logger.info("GENERATING_EMBEDDING_DISTANCE_FEATURES")
            emb_distances = compute_embedding_distances(names_x, names_y)

            # Absolute string length diff
            len_diffs = [
                abs(len(name_x) - len(name_y))
                for name_x, name_y in zip(names_x, names_y)
            ]

            df_featured = pd.DataFrame(
                {
                    self.jaccard_sim_col: jaccard_xy,
                    self.cosine_sim_col: cosine_sims_xy,
                    self.ratio_col: ratio_features,
                    self.sorted_token_ratio_col: sorted_token_ratios,
                    self.token_set_ratio_col: token_set_ratios,
                    self.partial_ratio_col: partial_ratios,
                    self.emb_dist_col: emb_distances,
                    self.len_diff_col: len_diffs,
                }
            )
            return df_featured
        except AssertionError as err:
            print(err)
            self.logger.error(err)
            return None

    def create_tfidf_vectorizer(self, all_names: List[str]) -> TfidfVectorizer:
        """
        Creates an n-gram TF-IDF vectorizer object to be used for generating TF-IDF vectors.

        :param all_names: The corpus (set of all person and orga names)
        :returns: The vectorizer object
        """

        try:
            assert len(all_names) > 0, "Empty corpus passed!"

            # Create a shared TF-IDF vectorizer
            self.logger.info("GENERATING_TF_IDF_VECTORIZER")
            tfidf_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_df=0.9, max_features=10000
            )
            tfidf_vectorizer.fit(all_names)

            # Save as pickle during training and load for inference
            filename_out = config["MODELPATH"]["FILENAME_MODEL_TFIDF_NGRAM"]
            with open(filename_out, "wb") as f:
                pickle.dump(tfidf_vectorizer, f)
            self.logger.info("SAVED_MODEL_TO", model="TF-IDF", file=filename_out)

            return tfidf_vectorizer
        except AssertionError as err:
            self.logger.error(err)
            return None

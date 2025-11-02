import time
import pickle
import warnings
import structlog

import pandas as pd
import seaborn as sns

from datetime import datetime
from typing import Any, List

from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from name_matching.config import read_config
from name_matching.log.logging import configure_structlog
from name_matching.features.build_features import FeatureGenerator
from name_matching.utils.cli import basic_argparser
from name_matching.utils.utils import plot_roc_auc, plot_precision_recall_auc, process_text_standard

config = read_config()
style.use("fivethirtyeight")

# Suppress pandas warnings
warnings.filterwarnings("ignore")


class NameMatchingTrainer:
    """
    Class to train the Name Matching ML model.
    """

    def __init__(
        self,
        logger: Any,
        test_size: float,
        thresh: float,
        df_train: pd.DataFrame,
        features_final: List[str],
        human_readable: bool,
    ) -> None:
        """
        Inits the class.

        :param logger: The logging object
        :param test_size: Size (fraction) of the test set
        :param thresh: Threshold for prediction (of the positive class)
        :param df_train: The featurized training data
        :param features_final: List of final features
        """

        self.logger = logger if logger is not None else structlog.get_logger()
        self.test_size = test_size
        self.thresh = thresh
        self.df_train = df_train
        self.features_final = features_final
        self.human_readable = human_readable

        self.label_col = config["DATA.COLUMNS"]["LABEL_COL"]
        self.figure_roc_auc = config["FIGUREPATH"]["FIGURE_ROC_AUC_TRAIN"]
        self.figure_precision_recall = config["FIGUREPATH"]["FIGURE_PRECISION_RECALL"]
        self.figure_feature_importance = config["FIGUREPATH"][
            "FIGURE_FEATURE_IMPORTANCE"
        ]
        self.figure_feature_distribution = config["FIGUREPATH"][
            "FIGURE_FEATURE_DISTRIBUTION"
        ]

    def train_model(self) -> LGBMClassifier:
        """
        Trains the Name Matching classification model.

        :return: The trained classifier
        """

        # Create training and test set
        x_train, x_test, y_train, y_test = train_test_split(
            self.df_train[self.features_final],
            self.df_train[self.label_col],
            test_size=self.test_size,
        )
        self.logger.info(
            "TRAIN_TEST_DF",
            x_train=x_train.shape,
            x_test=x_test.shape,
            y_train=y_train.shape,
            y_test=y_test.shape,
        )

        # Define a model and fit params
        hyper_params = {
            "num_leaves": 8,
            "max_depth": 4,
            "n_jobs": 4,
            "n_estimators": 4000,
            "learning_rate": 0.025,
            "is_unbalance": True,
        }
        model = LGBMClassifier(**hyper_params)

        fit_params = {
            # "early_stopping_rounds": 200,
            "eval_metric": "logloss",
            "eval_set": [(x_test, y_test)],
            # "verbose": False,
            "feature_name": "auto",
            "categorical_feature": "auto",
        }
        # Train the model
        model.fit(x_train, y_train, **fit_params)

        # The predicted True class given the cutoff threshold
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        y_pred = [1 if y > self.thresh else 0 for y in y_pred_prob]

        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info("TEST_ACCURACY", accuracy=round(accuracy * 100, 2))
        if self.human_readable:
            self.logger.info("CLASSIFICATION_REPORT", report="see below")
            print(classification_report(y_test, y_pred))
        else:
            self.logger.info(
                "CLASSIFICATION_REPORT",
                report=classification_report(y_test, y_pred, output_dict=True),
            )

        # Plot the ROC, PR curves
        self.plot_model(
            model, x_train, y_test, y_pred_prob
        )

        return model

    def plot_feature_distributions(self) -> None:
        """
        Plots the distributions of the features in training data.
        """

        fig = plt.figure(figsize=(16, 10))
        axs = [
            fig.add_subplot(2, int(len(self.features_final) * 0.5), i + 1)
            for i in range(len(self.features_final))
        ]

        for i, feature in enumerate(self.features_final):
            sns.distplot(
                self.df_train[self.df_train[self.label_col] == 1][feature],
                ax=axs[i],
                color="darkgreen",
                label="1",
            )
            sns.distplot(
                self.df_train[self.df_train[self.label_col] == 0][feature],
                ax=axs[i],
                color="darkorange",
                label="0",
            )
            axs[i].legend(fontsize=15)

        # plt.show()
        plt.savefig(self.figure_feature_distribution, bbox_inches="tight")
        plt.close()
        self.logger.info(
            "SAVED_PRECISION_RECALL_FIG_TO", file=self.figure_feature_distribution
        )

    def plot_model(
        self,
        model: LGBMClassifier,
        x_train: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred_prob: List[float],
    ) -> None:
        """
        Plots the trained model's performances on the test set.

        :param model: The trained model
        :param x_train: Training set (featured)
        :param y_test: Test set (labels)
        :param y_pred_prob: Predicted probabilities
        """

        # Plot the ROC curve
        plot_roc_auc(y_test, y_pred_prob, filename_out=self.figure_roc_auc)
        self.logger.info("SAVED_ROC_AUC_FIG_TO", file=self.figure_roc_auc)

        # Plot the PR curve
        pr_auc = plot_precision_recall_auc(y_test, y_pred_prob, filename_out=self.figure_precision_recall)
        self.logger.info("PRECISION_RECALL_AUC", pr_auc=round(pr_auc, 2))
        self.logger.info("SAVED_PRECISION_RECALL_FIG_TO", file=self.figure_precision_recall)

        # Plot the feature importance
        feature_importance = pd.DataFrame(
            sorted(zip(model.feature_importances_, x_train.columns)),
            columns=["Value", "Feature"],
        )
        sns.barplot(
            x="Value",
            y="Feature",
            data=feature_importance.sort_values(by="Value", ascending=False),
        )
        plt.title("Model Feature Importances")
        plt.savefig(self.figure_feature_importance, bbox_inches="tight")
        plt.close()
        self.logger.info(
            "SAVED_FEATURE_IMPORTANCE_TO", file=self.figure_feature_importance
        )


def main():
    """Run train model script"""
    parser = basic_argparser()
    parser.description = "Training Pipeline for Name Matching"
    parser.allow_abbrev = True
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--test-size",
        help="Size (fractional) of test size",
        default=0.2,
        required=False,
        type=float,
    )
    optional.add_argument(
        "--thresh",
        help="Threshold for prediction (of positive class)",
        default=0.85,
        required=False,
        type=float,
    )
    args = parser.parse_args()
    configure_structlog(args.silent)
    logger = structlog.get_logger()
    logger.info("TRAIN_MODEL_RUN_SCRIPT", **vars(args))

    # Start clocking the time
    start_time = time.time()

    # Load the filenames
    filename_train_featured = config["MODELPATH"]["FILENAME_TRAIN_FEATURED"]
    filename_pos_pairs = config["DATAPATH.PROCESSED"]["FILENAME_POS_PAIRS"]
    filename_neg_pairs = config["DATAPATH.PROCESSED"]["FILENAME_NEG_PAIRS"]

    # Load the env vars
    name_x_col = config["DATA.COLUMNS"]["NAME_X_COL"]
    name_y_col = config["DATA.COLUMNS"]["NAME_Y_COL"]
    label_col = config["DATA.COLUMNS"]["LABEL_COL"]
    datetime_col = config["DATA.COLUMNS"]["DATETIME_COL"]

    # Feature columns
    jaccard_sim_col = config["DATA.COLUMNS"]["JACCARD_SIM_COL"]
    cosine_sim_col = config["DATA.COLUMNS"]["COSINE_SIM_COL"]
    ratio_col = config["DATA.COLUMNS"]["RATIO_COL"]
    sorted_token_ratio_col = config["DATA.COLUMNS"]["SORTED_TOKEN_RATIO_COL"]
    token_set_ratio_col = config["DATA.COLUMNS"]["TOKEN_SET_RATIO_COL"]
    partial_ratio_col = config["DATA.COLUMNS"]["PARTIAL_RATIO_COL"]
    emb_dist_col = config["DATA.COLUMNS"]["EMB_DISTANCE_COL"]

    # Load the training data
    logger.info("LOAD_POSITIVE_PAIRS", file=filename_pos_pairs)
    df_pos_pairs = pd.read_csv(filename_pos_pairs)

    logger.info("LOAD_NEGATIVE_PAIRS", file=filename_neg_pairs)
    df_neg_pairs = pd.read_csv(filename_neg_pairs)

    # Filter the training data
    df_pos_pairs.dropna(subset=[name_x_col, name_y_col], inplace=True)
    logger.info("NON_NA_POSITIVE_PAIRS", df="df_pos_pairs", shape=df_pos_pairs.shape)
    df_pos_pairs.drop_duplicates(subset=[name_x_col, name_y_col], inplace=True)
    logger.info("DEDUPLICATED_POS_PAIRS", df="df_pos_pairs", shape=df_pos_pairs.shape)

    df_neg_pairs.dropna(subset=[name_x_col, name_y_col], inplace=True)
    logger.info("NON_NA_NEGATIVE_PAIRS", df="df_neg_pairs", shape=df_neg_pairs.shape)
    df_neg_pairs.drop_duplicates(subset=[name_x_col, name_y_col], inplace=True)
    logger.info("DEDUPLICATED_NEG_PAIRS", df="df_neg_pairs", shape=df_neg_pairs.shape)

    # Upper case all names
    logger.info("PREPROCESSING_NAMES_IN_TRAINING_DATA")
    df_pos_pairs[name_x_col] = df_pos_pairs[name_x_col].str.upper()
    df_pos_pairs[name_y_col] = df_pos_pairs[name_y_col].str.upper()
    df_neg_pairs[name_x_col] = df_neg_pairs[name_x_col].str.upper()
    df_neg_pairs[name_y_col] = df_neg_pairs[name_y_col].str.upper()
    
    # Pre-process the names (lowercase, strip, remove special chars, etc.)
    df_pos_pairs[name_x_col] = [process_text_standard(name, remove_stopwords=False) for name in df_pos_pairs[name_x_col]]
    df_pos_pairs[name_y_col] = [process_text_standard(name, remove_stopwords=False) for name in df_pos_pairs[name_y_col]]
    df_neg_pairs[name_x_col] = [process_text_standard(name, remove_stopwords=False) for name in df_neg_pairs[name_x_col]]
    df_neg_pairs[name_y_col] = [process_text_standard(name, remove_stopwords=False) for name in df_neg_pairs[name_y_col]]
    
    # Create a feature generator object
    generator = FeatureGenerator(logger)

    # Create a corpus of all unique names (persons and orgas)
    all_names = list(set(df_pos_pairs[name_x_col]))
    logger.info("TOTAL_NUM_UNIQUE_NAMES", count=len(all_names))
    tfidf_vectorizer = generator.create_tfidf_vectorizer(all_names)
    if tfidf_vectorizer is None:
        logger.error("UNEXPECTED_PIPELINE_TERMINATION")
        raise Exception("UNEXPECTED_PIPELINE_TERMINATION")

    # Feature engineering (positive)
    df_featured_pos = generator.build_features(
        df_pos_pairs[name_x_col].tolist(), df_pos_pairs[name_y_col].tolist(), tfidf_vectorizer
    )
    if df_featured_pos is None:
        logger.error("UNEXPECTED_PIPELINE_TERMINATION")
        raise Exception("UNEXPECTED_PIPELINE_TERMINATION")

    # Label the training data (positive)
    df_featured_pos[label_col] = 1
    logger.info(
        "POSITIVE_FEATURE_DF", df="df_featured_pos", shape=df_featured_pos.shape
    )

    # Build the features (negative)
    df_featured_neg = generator.build_features(
        df_neg_pairs[name_x_col].tolist(), df_neg_pairs[name_y_col].tolist(), tfidf_vectorizer
    )
    if df_featured_neg is None:
        logger.error("UNEXPECTED_PIPELINE_TERMINATION")
        raise Exception("UNEXPECTED_PIPELINE_TERMINATION")

    # Label the training data (negative)
    df_featured_neg[label_col] = 0
    logger.info(
        "NEGATIVE_FEATURE_DF", df="df_featured_neg", shape=df_featured_neg.shape
    )

    # Combine the positive and negative featured data frames
    df_train = pd.concat([df_featured_pos, df_featured_neg], ignore_index=True)
    logger.info("FEATURE_DF", df="df_train", shape=df_train.shape)

    # Insert timestamp of the training
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    df_train[datetime_col] = date_time

    # Store the featured data frame in DVC for model monitoring
    df_train.to_csv(filename_train_featured, index=False)
    logger.info("SAVED_FEATURED_TRAINING_DATA_TO", file=filename_train_featured)

    # The final list of features used for training
    features_final = [
        jaccard_sim_col,
        cosine_sim_col,
        ratio_col,
        sorted_token_ratio_col,
        token_set_ratio_col,
        partial_ratio_col,
        emb_dist_col
    ]
    logger.info("NUM_FINAL_FEATURES", count=len(features_final))

    # Create a trainer object
    trainer = NameMatchingTrainer(
        logger,
        args.test_size,
        args.thresh,
        df_train,
        features_final,
        args.human_readable,
    )
    # Plot the feature distributions
    # trainer.plot_feature_distributions()

    # Train the model
    model = trainer.train_model()

    # Save the trained model to disk
    filename_out = config["MODELPATH"]["MODEL_LGB_NAME_MATCHING"]
    with open(filename_out, "wb") as f:
        pickle.dump(model, f)
    logger.info("SAVED_MODEL_TO", model="lgb", file=filename_out)

    # Compute the time elapsed
    end_time = time.time()
    time_taken = float(end_time - start_time) / 60
    logger.info("TOTAL_RUNTIME", time_taken=round(time_taken, 4), unit="minutes")


if __name__ == "__main__":
    main()

import os
import time
import random
import warnings
import structlog
import editdistance

import pandas as pd

from tqdm import tqdm
from pathlib import Path

from openai import OpenAI
from mistralai import Mistral
from dotenv import load_dotenv

from configparser import ConfigParser
from typing import Any, Dict, List, Tuple

from name_matching.config import read_config
from name_matching.log.logging import configure_structlog, configure_tqdm
from name_matching.utils.cli import basic_argparser
from name_matching.utils.utils import (
    generate_typo_name,
    process_text_standard,
    generate_aliases
)

# Instantiate configuration class
config = read_config()

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Load the environment variables
load_dotenv()

OPENAI_API_TOKEN = os.environ["OPENAI_API_KEY"]

# AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
# AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
# AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
# AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]


class TrainingDataGenerator:
    """
    Class for training data (positive and negative examples) generation.
    """

    def __init__(
        self,
        n_neg: int,
        logger: Any,
        config: ConfigParser,
    ) -> None:
        """
        Inits the TrainingDataGenerator class.

        :param n_neg: Number of negative examples
        :param logger: The logging object
        :param config: The config parser
        """

        # Set the instance variables
        self.n_neg = n_neg
        self.logger = logger if logger is not None else structlog.get_logger()
        self.config = config

        # Define the column names
        self.first_name_col = self.config["DATA.COLUMNS"]["FIRST_NAME_COL"]
        self.last_name_col = self.config["DATA.COLUMNS"]["LAST_NAME_COL"]
        self.full_name_col = self.config["DATA.COLUMNS"]["FULL_NAME_COL"]
        self.ent_type_col = self.config["DATA.COLUMNS"]["ENT_TYPE_COL"]
        self.edit_dist_col = self.config["DATA.COLUMNS"]["EDIT_DIST_COL"]
        self.name_x_col = self.config["DATA.COLUMNS"]["NAME_X_COL"]
        self.name_y_col = self.config["DATA.COLUMNS"]["NAME_Y_COL"]


    def load_training_data(self) -> pd.DataFrame:
        """
        Loads the training data.

        :return: Training data frame
        """

        # Load the raw training data
        filename_train = self.config["DATAPATH.RAW"]["MOCK_SAMPLE"]
        self.logger.info("LOADING_RAW_TRAINING_DATA", file=filename_train)

        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        self.logger.info("LOAD_RAW_TRAINING_DATA", source="local")
        self.df_train = pd.read_csv(filename_train)

        # Initial filtering
        self.df_train.dropna(subset=[self.full_name_col], inplace=True)
        self.df_train.drop_duplicates(subset=[self.full_name_col, self.ent_type_col], inplace=True)
        return self.df_train

    def normalize_name_strings(self) -> pd.DataFrame:
        """
        Makes all the name strings consistent.

        :return: Training data frame
        """

        # Upper case all names
        full_names = [text.upper() for text in self.df_train[self.full_name_col]]
        first_names = [text.upper() for text in self.df_train[self.first_name_col]]
        last_names = [text.upper() for text in self.df_train[self.last_name_col]]

        # Standardize the names
        normalized_full_names = [
            process_text_standard(text, remove_stopwords=False) for text in full_names
        ]
        normalized_first_names = [
            process_text_standard(text, remove_stopwords=False) for text in first_names
        ]
        normalized_last_names = [
            process_text_standard(text, remove_stopwords=False) for text in last_names
        ]
        
        self.df_train[self.full_name_col] = normalized_full_names
        self.df_train[self.first_name_col] = normalized_first_names
        self.df_train[self.last_name_col] = normalized_last_names

        return self.df_train


    def shorten_names(self, df_train: pd.DataFrame) -> List[str]:
        """
        Shortens the full names for positive example generation.

        :param df_train: Training data frame
        :return: List of modified names
        """

        pass

    def abbreviate_orga_types(self, df_train: pd.DataFrame) -> List[str]:
        """
        Abbreviates (or unabbreviates) organization types (e.g., Ltd <-> Limited, Pte <-> Private)

        :param df_train: Training data frame for organization entities
        :return: List of orga names whose types are abbreviated (or vice versa)
        """

        pass

    def remove_orga_types(self, df_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Removes organization types (e.g., Ltd, Inc, Corp, AG, SA, etc.)

        :param df_train: Training data frame for organization entities
        :return: Tuple of list of strings for orgas and extracted types
        """

        pass

    def generate_pos_mappings(
        self, df_train: pd.DataFrame, is_person: bool = True
    ) -> Dict[str, List[str]]:
        """
        Generates mappings for positive examples (first parties).

        :param df_train: Training data frame
        :param is_person: Whether it's for person or organization entities
        :return: Dictionary of mappings
        """

        if is_person:
            pos_aliases = [
                self.full_name_col,
                self.reversed_full_name_col,
                self.permuted_full_name_col,
                self.abbrev_first_name_col,
                self.titled_name_col,
                self.titled_full_name_col,
                self.shortened_name_col,
                self.typo_name_col,
            ]
        else:
            # Organization
            pos_aliases = [
                self.full_name_col,
                self.abbrev_orga_type_col,
                self.orga_type_truncated_col,
                self.typo_name_col,
            ]
        pos_mapping = {}
        for i in tqdm(range(len(df_train))):
            full_name = df_train[self.full_name_col][i]
            aliases = [df_train[alias][i] for alias in pos_aliases]
            pos_mapping[full_name] = aliases

        return pos_mapping

    def generate_df_pairs(self, mappings: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Generates data frame of pairs of names.

        :param mappings: Name mappings
        :return: Data frame of pairs of names
        """

        # Create mapping pairs
        list_keys = []
        list_vals = []
        for key in tqdm(mappings.keys()):
            vals = mappings[key]
            list_keys += [key] * len(vals)
            list_vals += vals
        df_pairs = pd.DataFrame(
            {self.name_x_col: list_keys, self.name_y_col: list_vals}
        )

        return df_pairs

    def generate_neg_mappings(
        self, df_train: pd.DataFrame, df_sample: pd.DataFrame, is_person: bool = True
    ) -> Dict[str, List[str]]:
        """
        Generates mappings for negative examples (third parties).

        :param df_train: Training data frame
        :param df_sample: Data frame to sample negative examples from
        :param is_person: Whether it's for person or organization entities
        :return: Dictionary of mappings
        """

        # List of aliases
        if is_person:
            pos_aliases = [
                self.full_name_col,
                self.reversed_full_name_col,
                self.permuted_full_name_col,
                self.abbrev_first_name_col,
                self.titled_name_col,
                self.titled_full_name_col,
                self.shortened_name_col,
                self.typo_name_col,
            ]
        else:
            # Organization
            pos_aliases = [
                self.full_name_col,
                self.abbrev_orga_type_col,
                self.orga_type_truncated_col,
                self.typo_name_col,
            ]

        neg_mapping = {}
        for i in tqdm(range(len(df_train))):
            full_name = df_train[self.full_name_col][i]
            first_name = df_train[self.first_name_col][i]
            last_name = df_train[self.last_name_col][i]

            # The set to sample negative indices from
            df_sampled = df_sample[df_sample[self.full_name_col] != full_name]
            sample_set = set(df_sampled.index)

            # Negative examples having the same first/last names
            df_sampled_same_first = df_sampled[
                df_sampled[self.first_name_col] == first_name
            ]
            df_sampled_same_last = df_sampled[
                df_sampled[self.last_name_col] == last_name
            ]
            
            if len(df_sampled_same_first) > 0 or len(df_sampled_same_last) > 0:
                df_sampled_same = pd.concat(
                    [df_sampled_same_last, df_sampled_same_first], ignore_index=True
                )
                sample_set -= set(df_sampled_same.index)
            else:
                df_sampled_same = pd.DataFrame()
            # Sample the negative indices
            num_samples = self.n_neg * 5
            if len(sample_set) > num_samples:
                sampled_idx = random.sample(sample_set, num_samples)
            else:
                sampled_idx = list(sample_set)

            # The sampled data frame
            if len(sampled_idx) > 0:
                df_sampled = df_sample.loc[sampled_idx]
                # Sort df_sampled by edit distance with the full name
                sampled_edit_dist = [
                    editdistance.eval(full_name, name_y)
                    for name_y in df_sampled[self.full_name_col]
                ]
                df_sampled[self.edit_dist_col] = sampled_edit_dist
                df_sampled.sort_values(
                    by=self.edit_dist_col,
                    ascending=True,
                    inplace=True,
                    ignore_index=True,
                )
                # Take the top (nearest negative examples)
                df_sampled = df_sampled.head(self.n_neg)
                # Adding those with same first/last names
                if len(df_sampled_same) > 0:
                    df_sampled = pd.concat(
                        [df_sampled_same, df_sampled], ignore_index=True
                    )
                    df_sampled = df_sampled.head(self.n_neg)
                list_sampled_neg = []
                for j in range(len(df_sampled)):
                    # Sample one of the aliases as negative example
                    alias_choice = random.choice(pos_aliases)
                    sampled_neg = df_sampled[alias_choice][j]
                    list_sampled_neg.append(sampled_neg)
                neg_mapping[full_name] = list_sampled_neg

        return neg_mapping


def main():
    """Run make dataset script"""
    parser = basic_argparser()
    parser.description = "Training Data Generation for Name Matching Model"
    parser.allow_abbrev = True
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--n_neg",
        help="Number of negative examples",
        default=10,
        required=False,
        type=int,
    )

    args = parser.parse_args()
    configure_structlog(args.silent)
    configure_tqdm(args.disable_tqdm)
    logger = structlog.get_logger()
    logger.info("MAKE_DATASET_RUN_SCRIPT", **vars(args))

    # Start clocking the time
    start_time = time.time()
    logger.info("TRAINING_DATA_GENERATION_FOR_NAME_MATCHING")

    # Define column names and values
    first_name_col = config["DATA.COLUMNS"]["FIRST_NAME_COL"]
    last_name_col = config["DATA.COLUMNS"]["LAST_NAME_COL"]
    full_name_col = config["DATA.COLUMNS"]["FULL_NAME_COL"]
    ent_type_col = config["DATA.COLUMNS"]["ENT_TYPE_COL"]
    
    person_type_val = config["DATA.VALUES"]["PERSON_TYPE"]
    orga_type_val = config["DATA.VALUES"]["ORGA_TYPE"]
    filename_pos_pairs = config["DATAPATH.PROCESSED"]["FILENAME_POS_PAIRS"]
    filename_neg_pairs = config["DATAPATH.PROCESSED"]["FILENAME_NEG_PAIRS"]

    # Create a training data generator object
    generator = TrainingDataGenerator(n_neg=args.n_neg, logger=logger, config=config)

    # Load the training data
    df_train = generator.load_training_data()
    logger.info("TRAINING_DATA_DF", df="df_train", shape=df_train.shape)

    # df_train = generator.normalize_name_strings()
    # logger.info("NORMALIZED_NAMES_DATA_DF", df="df_train", shape=df_train.shape)
    
    # Separate person and organization entities
    df_train_person = df_train[df_train[ent_type_col] == person_type_val]
    logger.info(
        "PERSON_TRAINING_DATA_DF", df="df_train_person", shape=df_train_person.shape
    )

    df_train_orga = df_train[df_train[ent_type_col] == orga_type_val]
    logger.info("ORGA_TRAINING_DATA_DF", df="df_train_orga", shape=df_train_orga.shape)

    # Process person entities
    logger.info("GENERATING_POSITIVE_ALIASES_FOR_PERSON_ENTITIES")
    # Read the system prompt
    prompt_file = "data/pers_alias_prompt.txt"
    sys_prompt = None
    with open(prompt_file, 'r', encoding='utf-8') as file:
        sys_prompt = file.read()

    if sys_prompt is None or len(sys_prompt) == 0:
        logger.error("UNEXPECTED_PIPELINE_TERMINATION")
        raise Exception("UNEXPECTED_PIPELINE_TERMINATION")
    
    # Initialize the client
    client = OpenAI(api_key=OPENAI_API_TOKEN)
    

    typo_names = [
        generate_typo_name(name, prob_flip=0.4)
        for name in df_train_person[full_name_col]
    ]
    print(typo_names[:5])
    
    logger.info("GENERATING_POSITIVE_ALIASES_FOR_ORGA")
    df_train_orga = generator.process_orga_entities(df_train_orga)
    orgas_trunc_types, orga_types = generator.remove_orga_types(df_train_orga)
    # Apparently, orga_types are not used for anything
    orgas_typos = [
        generate_typo_name(name, prob_flip=0.35)  # TODO: extract var?
        for name in df_train_orga[full_name_col]
    ]
    # Abbreviate and unabbreviate orga type as positive aliases
    df_train_orga[abbrev_orga_type_col] = generator.abbreviate_orga_types(df_train_orga)
    # Other strategies for orga-typed aliases
    df_train_orga[orga_type_truncated_col] = orgas_trunc_types
    df_train_orga[typo_name_col] = orgas_typos
    df_train_orga[orga_type_col] = orga_types
    logger.info("GENERATING_POSITIVE_ALIASES_FOR_HYPHENED_ORGA")
    df_orga_hyphens = df_train_orga[df_train_orga[has_hyphen_col]]
    logger.info("HYPHENED_ORGA_DF", df="df_orga_hyphens", shape=df_orga_hyphens.shape)
    orig_full_names = df_orga_hyphens[full_name_col].tolist()
    orig_abbrev_names = df_orga_hyphens[abbrev_orga_type_col].tolist()
    orgas_trunc_types, orga_types = generator.remove_orga_types(df_orga_hyphens)
    df_orga_hyphens[full_name_col] = df_orga_hyphens[dehyphened_last_name_col]
    df_orga_hyphens[abbrev_orga_type_col] = orig_abbrev_names
    df_orga_hyphens[orga_type_truncated_col] = orgas_trunc_types
    df_orga_hyphens[typo_name_col] = orig_full_names
    df_orga_hyphens[orga_type_col] = orga_types
    # Reset all the indices
    df_person_single.reset_index(drop=True, inplace=True)
    df_single_hyphens.reset_index(drop=True, inplace=True)
    df_train_orga.reset_index(drop=True, inplace=True)
    df_orga_hyphens.reset_index(drop=True, inplace=True)
    logger.info("GENERATING_POSITIVE_MAPPINGS")
    single_person_pos_mappings = generator.generate_pos_mappings(
        df_person_single, is_person=True
    )
    hyphened_person_pos_mappings = generator.generate_pos_mappings(
        df_single_hyphens, is_person=True
    )
    # Separate joint accounts into those that are well-tokenized (negative examples) and not (positive examples)
    len_joints = [
        [len(token) for token in joints] for joints in df_person_joint[joint_col]
    ]
    df_person_joint[len_joint_col] = len_joints
    has_empty_joints = [0 in len_list for len_list in len_joints]
    df_person_joint[has_empty_joint_col] = has_empty_joints
    # Positive examples -- not well-tokenized (there is an empty token)
    df_person_joint_pos = df_person_joint[df_person_joint[has_empty_joint_col]]
    # Negative examples -- well-tokenized (AH names are well-separated)
    df_person_joint_neg = df_person_joint[~df_person_joint[has_empty_joint_col]]
    logger.info(
        "JOINT_PERSONS_POSITIVE_EXAMPLES_DF",
        df="df_person_joint_pos",
        shape=df_person_joint_pos.shape,
    )
    logger.info(
        "JOINT_PERSONS_NEGATIVE_EXAMPLES_DF",
        df="df_person_joint_neg",
        shape=df_person_joint_neg.shape,
    )

    # Positive examples (of joint accounts)
    if len(df_person_joint_pos) > 0:
        joint_person_pos_mappings = dict(
            zip(df_person_joint_pos[full_name_col], df_person_joint_pos[joint_col])
        )
    else:
        joint_person_pos_mappings = dict()

    # Negative examples (of joint accounts)
    if len(df_person_joint_neg) > 0:
        joint_person_neg_mappings = dict(
            zip(df_person_joint_neg[full_name_col], df_person_joint_neg[joint_col])
        )
    else:
        joint_person_neg_mappings = dict()

    orga_pos_mappings = generator.generate_pos_mappings(df_train_orga, is_person=False)
    hyphened_orga_pos_mappings = generator.generate_pos_mappings(
        df_orga_hyphens, is_person=False
    )
    # For person (positive pairs generation)
    df_pos_person_single = generator.generate_df_pairs(single_person_pos_mappings)
    df_pos_person_single[has_joint_x_col] = False
    df_pos_person_single[has_joint_y_col] = False
    df_pos_person_single[orga_type_x_col] = False
    df_pos_person_single[orga_type_y_col] = False
    logger.info(
        "SINGLE_PERSONS_POSITIVE_PAIRS_DF",
        df="df_pos_person_single",
        shape=df_pos_person_single.shape,
    )
    df_pos_person_hyphened = generator.generate_df_pairs(hyphened_person_pos_mappings)
    df_pos_person_hyphened[has_joint_x_col] = False
    df_pos_person_hyphened[has_joint_y_col] = False
    df_pos_person_hyphened[orga_type_x_col] = False
    df_pos_person_hyphened[orga_type_y_col] = False
    logger.info(
        "HYPHENED_PERSONS_POSITIVE_PAIRS_DF",
        df="df_pos_person_hyphened",
        shape=df_pos_person_hyphened.shape,
    )

    if len(joint_person_pos_mappings) > 0:
        df_pos_person_joint = generator.generate_df_pairs(joint_person_pos_mappings)
        df_pos_person_joint[has_joint_x_col] = True
        df_pos_person_joint[has_joint_y_col] = True
        df_pos_person_joint[orga_type_x_col] = False
        df_pos_person_joint[orga_type_y_col] = False
        logger.info(
            "JOINT_PERSONS_POSITIVE_PAIRS_DF",
            df="df_pos_person_joint",
            shape=df_pos_person_joint.shape,
        )
    else:
        df_pos_person_joint = pd.DataFrame()

    # For organization (positive pairs generation)
    df_pos_orga = generator.generate_df_pairs(orga_pos_mappings)
    df_pos_orga[has_joint_x_col] = False
    df_pos_orga[has_joint_y_col] = False
    df_pos_orga[orga_type_x_col] = True
    df_pos_orga[orga_type_y_col] = True
    logger.info("ORGA_POSITIVE_PAIRS_DF", df="df_pos_orga", shape=df_pos_orga.shape)
    df_pos_orga_hyphened = generator.generate_df_pairs(hyphened_orga_pos_mappings)
    df_pos_orga_hyphened[has_joint_x_col] = False
    df_pos_orga_hyphened[has_joint_y_col] = False
    df_pos_orga_hyphened[orga_type_x_col] = True
    df_pos_orga_hyphened[orga_type_y_col] = True
    logger.info(
        "HYPHENED_ORGA_POSITIVE_PAIRS_DF",
        df="df_pos_orga_hyphened",
        shape=df_pos_orga_hyphened.shape,
    )
    # Combine both (persons and organizations)
    df_pos_person = pd.concat(
        [df_pos_person_joint, df_pos_person_single, df_pos_person_hyphened],
        ignore_index=True,
    )
    df_pos_orga = pd.concat([df_pos_orga, df_pos_orga_hyphened], ignore_index=True)
    df_pairs_pos = pd.concat([df_pos_person, df_pos_orga], ignore_index=True)
    logger.info("POSITIVE_PAIRS_DF", df="df_pairs_pos", shape=df_pairs_pos.shape)
    df_pairs_pos.to_csv(filename_pos_pairs, index=False)
    logger.info("SAVE_POSITIVE_PAIRS", file=filename_pos_pairs)
    logger.info("GENERATING_NEGATIVE_MAPPINGS")
    # For person (mapping generation)
    person_person_neg_mappings = generator.generate_neg_mappings(
        df_person_single, df_person_single
    )
    person_orga_neg_mappings = generator.generate_neg_mappings(
        df_person_single,
        df_train_orga,
        is_person=False,
    )
    person_hyphen_mappings = generator.generate_neg_mappings(
        df_single_hyphens, df_single_hyphens
    )
    # For organization (mapping generation)
    orga_person_neg_mappings = generator.generate_neg_mappings(
        df_train_orga, df_person_single
    )
    orga_orga_neg_mappings = generator.generate_neg_mappings(
        df_train_orga, df_train_orga, is_person=False
    )
    orga_hyphen_neg_mappings = generator.generate_neg_mappings(
        df_orga_hyphens, df_orga_hyphens, is_person=False
    )
    # For person (negative pairs generation)
    if len(joint_person_neg_mappings) > 0:
        df_neg_person_joint = generator.generate_df_pairs(joint_person_neg_mappings)
        df_neg_person_joint[has_joint_x_col] = True
        df_neg_person_joint[has_joint_y_col] = False
        df_neg_person_joint[orga_type_x_col] = False
        df_neg_person_joint[orga_type_y_col] = False
        logger.info(
            "JOINT_PERSONS_NEGATIVE_PAIRS_DF",
            df="df_neg_person_joint",
            shape=df_neg_person_joint.shape,
        )
    else:
        df_neg_person_joint = pd.DataFrame()

    df_neg_person_person = generator.generate_df_pairs(person_person_neg_mappings)
    df_neg_person_person[has_joint_x_col] = False
    df_neg_person_person[has_joint_y_col] = False
    df_neg_person_person[orga_type_x_col] = False
    df_neg_person_person[orga_type_y_col] = False
    logger.info(
        "PERSON_PERSONS_NEGATIVE_PAIRS_DF",
        df="df_neg_person_person",
        shape=df_neg_person_person.shape,
    )
    df_neg_person_orga = generator.generate_df_pairs(person_orga_neg_mappings)
    df_neg_person_orga[has_joint_x_col] = False
    df_neg_person_orga[has_joint_y_col] = False
    df_neg_person_orga[orga_type_x_col] = False
    df_neg_person_orga[orga_type_y_col] = True
    logger.info(
        "ORGA_PERSONS_NEGATIVE_PAIRS_DF",
        df="df_neg_person_orga",
        shape=df_neg_person_orga.shape,
    )
    df_neg_person_hyphen = generator.generate_df_pairs(person_hyphen_mappings)
    df_neg_person_hyphen[has_joint_x_col] = False
    df_neg_person_hyphen[has_joint_y_col] = False
    df_neg_person_hyphen[orga_type_x_col] = False
    df_neg_person_hyphen[orga_type_y_col] = False
    logger.info(
        "HYPHENED_PERSONS_NEGATIVE_PAIRS_DF",
        df="df_neg_person_hyphen",
        shape=df_neg_person_hyphen.shape,
    )
    df_neg_person = pd.concat(
        [
            df_neg_person_joint,
            df_neg_person_person,
            df_neg_person_orga,
            df_neg_person_hyphen,
        ],
        ignore_index=True,
    )
    logger.info(
        "PERSONS_NEGATIVE_PAIRS_DF", df="df_neg_person", shape=df_neg_person.shape
    )
    # For organization (negative pairs generation)
    df_neg_orga_person = generator.generate_df_pairs(orga_person_neg_mappings)
    df_neg_orga_person[has_joint_x_col] = False
    df_neg_orga_person[has_joint_y_col] = False
    df_neg_orga_person[orga_type_x_col] = True
    df_neg_orga_person[orga_type_y_col] = False
    logger.info(
        "ORGA_PERSON_NEGATIVE_PAIRS_DF",
        df="df_neg_orga_person",
        shape=df_neg_orga_person.shape,
    )
    df_neg_orga_orga = generator.generate_df_pairs(orga_orga_neg_mappings)
    df_neg_orga_orga[has_joint_x_col] = False
    df_neg_orga_orga[has_joint_y_col] = False
    df_neg_orga_orga[orga_type_x_col] = True
    df_neg_orga_orga[orga_type_y_col] = True
    logger.info(
        "ORGA_ORGA_NEGATIVE_PAIRS_DF",
        df="df_neg_orga_orga",
        shape=df_neg_orga_orga.shape,
    )
    df_neg_orga_hyphen = generator.generate_df_pairs(orga_hyphen_neg_mappings)
    df_neg_orga_hyphen[has_joint_x_col] = False
    df_neg_orga_hyphen[has_joint_y_col] = False
    df_neg_orga_hyphen[orga_type_x_col] = True
    df_neg_orga_hyphen[orga_type_y_col] = True
    logger.info(
        "HYPHENED_ORGA_NEGATIVE_PAIRS_DF",
        df="df_neg_orga_hyphen",
        shape=df_neg_orga_hyphen.shape,
    )
    df_neg_orga = pd.concat(
        [df_neg_orga_person, df_neg_orga_orga, df_neg_orga_hyphen], ignore_index=True
    )
    logger.info("ORGA_NEGATIVE_PAIRS_DF", df="df_neg_orga", shape=df_neg_orga.shape)
    df_pairs_neg = pd.concat([df_neg_person, df_neg_orga], ignore_index=True)
    logger.info("NEGATIVE_PAIRS_DF", df="df_pairs_neg", shape=df_pairs_neg.shape)

    # Save to disk
    df_pairs_neg.to_csv(filename_neg_pairs, index=False)
    logger.info("SAVE_NEGATIVE_PAIRS", file=filename_neg_pairs)
    end_time = time.time()
    time_taken = float(end_time - start_time) / 60
    logger.info("TOTAL_RUNTIME", time_taken=round(time_taken, 4), unit="minutes")


if __name__ == "__main__":
    main()

import os
import time
import random
import warnings
import structlog
import editdistance

import pandas as pd

from tqdm import tqdm
from pathlib import Path

# from openai import OpenAI
from openai import AzureOpenAI
from dotenv import load_dotenv

from typing import Any, Dict, List
from configparser import ConfigParser

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

AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]


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

    def generate_pos_mappings(
        self, full_names: List[str], list_aliases: List[List[str]]
    ) -> Dict[str, List[str]]:
        """
        Generates mappings for positive examples (first parties).

        :param full_names: List of full names
        :param list_aliases: List of list of positive aliases
        :return: Dictionary of mappings
        """

        pos_mapping = {}
        for i in tqdm(range(len(full_names))):
            full_name = full_names[i]
            aliases = list_aliases[i]

            if full_name not in aliases:
                # Ensure the full name is part of the aliases
                aliases.insert(0, full_name)

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
        self, df_train: pd.DataFrame, df_sample: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generates mappings for negative examples (third parties).

        :param df_train: Training data frame
        :param df_sample: Data frame to sample negative examples from
        :return: Dictionary of mappings
        """

        neg_mapping = {}
        full_names = df_train[self.full_name_col].tolist()
        first_names = df_train[self.first_name_col].tolist()
        last_names = df_train[self.last_name_col].tolist()
        
        for full_name, first_name, last_name in tqdm(zip(full_names, first_names, last_names)):
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
                
                neg_mapping[full_name] = df_sampled[self.full_name_col].tolist()

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

    filename_orga_alias = config["DATAPATH.RAW"]["ORGA_ALIAS_PROMPT"]
    filename_pers_alias = config["DATAPATH.RAW"]["PERS_ALIAS_PROMPT"]
    filename_pos_pairs = config["DATAPATH.PROCESSED"]["FILENAME_POS_PAIRS"]
    filename_neg_pairs = config["DATAPATH.PROCESSED"]["FILENAME_NEG_PAIRS"]

    # Create a training data generator object
    generator = TrainingDataGenerator(n_neg=args.n_neg, logger=logger, config=config)

    # Load the training data
    df_train = generator.load_training_data()
    logger.info("TRAINING_DATA_DF", df="df_train", shape=df_train.shape)

    # Separate person and organization entities
    df_train_person = df_train[df_train[ent_type_col] == person_type_val]
    logger.info(
        "PERSON_TRAINING_DATA_DF", df="df_train_person", shape=df_train_person.shape
    )

    df_train_orga = df_train[df_train[ent_type_col] == orga_type_val]
    logger.info("ORGA_TRAINING_DATA_DF", df="df_train_orga", shape=df_train_orga.shape)

    logger.info("GENERATING_POSITIVE_ALIASES_FOR_PERSON_ENTITIES")
    # Read the system prompt
    sys_prompt = None
    with open(filename_pers_alias, 'r', encoding='utf-8') as file:
        sys_prompt = file.read()
    
    # Initialize the client
    # client = OpenAI(api_key=OPENAI_API_TOKEN)
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Iterate through all the names and generate aliases
    person_aliases = []
    for orga_name, first_name, last_name in tqdm(zip(df_train_person[full_name_col], df_train_person[first_name_col], df_train_person[last_name_col])):
        alias_str = generate_aliases(client, sys_prompt, full_name=orga_name, first_name=first_name, last_name=last_name, model=AZURE_OPENAI_DEPLOYMENT)
        aliases = [alias.strip() for alias in alias_str.split(';') if alias.strip()]
        person_aliases.append(aliases)        

    # Generate typo aliases
    person_typos = [
        generate_typo_name(name, prob_flip=0.4)
        for name in df_train_person[full_name_col]
    ]
    
    # Add typo names to the list of aliases
    for i in tqdm(range(len(person_aliases))):
        if person_typos[i] not in person_aliases[i]:
            person_aliases[i].append(person_typos[i])
    
    logger.info("GENERATING_POSITIVE_ALIASES_FOR_ORGANIZATION_ENTITIES")
    # Read the system prompt
    sys_prompt = None
    with open(filename_orga_alias, 'r', encoding='utf-8') as file:
        sys_prompt = file.read()
    
    # Iterate through all the names and generate aliases
    orga_aliases = []
    for orga_name in tqdm(df_train_orga[full_name_col].tolist()):
        alias_str = generate_aliases(client, sys_prompt, full_name=orga_name, model=AZURE_OPENAI_DEPLOYMENT)
        aliases = [alias.strip() for alias in alias_str.split(';') if alias.strip()]
        orga_aliases.append(aliases)

    # Generate typo aliases
    orga_typos = [
        generate_typo_name(name, prob_flip=0.35)
        for name in df_train_orga[full_name_col]
    ]
    
    # Add typo names to the list of aliases
    for i in tqdm(range(len(orga_aliases))):
        if orga_typos[i] not in orga_aliases[i]:
            print(orga_typos[i])
            orga_aliases[i].append(orga_typos[i])

    # Generate positive mappings
    logger.info("GENERATING_POSITIVE_MAPPINGS")
    person_pos_mappings = generator.generate_pos_mappings(
        df_train_person[full_name_col].tolist(), person_aliases
    )
    orga_pos_mappings = generator.generate_pos_mappings(
        df_train_orga[full_name_col].tolist(), orga_aliases
    )

    # Generate positive pairs data frames
    # For persons
    df_pos_person = generator.generate_df_pairs(person_pos_mappings)
    logger.info("PERSON_POSITIVE_PAIRS_DF", df="df_pos_person", shape=df_pos_person.shape)

    # For organizations
    df_pos_orga = generator.generate_df_pairs(orga_pos_mappings)
    logger.info("ORGA_POSITIVE_PAIRS_DF", df="df_pos_orga", shape=df_pos_orga.shape)
    
    # Combine both (persons and organizations)
    df_pairs_pos = pd.concat([df_pos_person, df_pos_orga], ignore_index=True)
    logger.info("POSITIVE_PAIRS_DF", df="df_pairs_pos", shape=df_pairs_pos.shape)
    df_pairs_pos.to_csv(filename_pos_pairs, index=False)
    logger.info("SAVED_POSITIVE_PAIRS", file=filename_pos_pairs)
    
    logger.info("GENERATING_NEGATIVE_MAPPINGS")
    # For person (mapping generation)
    person_person_neg_mappings = generator.generate_neg_mappings(
        df_train_person, df_train_person
    )
    person_orga_neg_mappings = generator.generate_neg_mappings(
        df_train_person,
        df_train_orga,
    )
    
    # For organization (mapping generation)
    orga_person_neg_mappings = generator.generate_neg_mappings(
        df_train_orga, df_train_person
    )
    orga_orga_neg_mappings = generator.generate_neg_mappings(
        df_train_orga, df_train_orga
    )
    
    # Generate negative pairs data frames (Person)
    df_neg_person_person = generator.generate_df_pairs(person_person_neg_mappings)
    logger.info(
        "PERSON_PERSON_NEGATIVE_PAIRS_DF",
        df="df_neg_person_person",
        shape=df_neg_person_person.shape,
    )

    df_neg_person_orga = generator.generate_df_pairs(person_orga_neg_mappings)
    logger.info(
        "ORGA_PERSON_NEGATIVE_PAIRS_DF",
        df="df_neg_person_orga",
        shape=df_neg_person_orga.shape,
    )

    df_neg_person = pd.concat(
        [
            df_neg_person_person,
            df_neg_person_orga,
        ],
        ignore_index=True,
    )
    logger.info(
        "PERSON_NEGATIVE_PAIRS_DF", df="df_neg_person", shape=df_neg_person.shape
    )

    # # Generate negative pairs data frames (Organization)
    df_neg_orga_person = generator.generate_df_pairs(orga_person_neg_mappings)
    logger.info(
        "ORGA_PERSON_NEGATIVE_PAIRS_DF",
        df="df_neg_orga_person",
        shape=df_neg_orga_person.shape,
    )

    df_neg_orga_orga = generator.generate_df_pairs(orga_orga_neg_mappings)
    logger.info(
        "ORGA_ORGA_NEGATIVE_PAIRS_DF",
        df="df_neg_orga_orga",
        shape=df_neg_orga_orga.shape,
    )

    df_neg_orga = pd.concat(
        [df_neg_orga_person, df_neg_orga_orga], ignore_index=True
    )
    logger.info("ORGA_NEGATIVE_PAIRS_DF", df="df_neg_orga", shape=df_neg_orga.shape)

    df_pairs_neg = pd.concat([df_neg_person, df_neg_orga], ignore_index=True)
    logger.info("NEGATIVE_PAIRS_DF", df="df_pairs_neg", shape=df_pairs_neg.shape)

    # Save to disk
    df_pairs_neg.to_csv(filename_neg_pairs, index=False)
    logger.info("SAVED_NEGATIVE_PAIRS", file=filename_neg_pairs)

    # End clocking the time
    end_time = time.time()
    time_taken = float(end_time - start_time) / 60
    logger.info("TOTAL_RUNTIME", time_taken=round(time_taken, 4), unit="minutes")


if __name__ == "__main__":
    main()

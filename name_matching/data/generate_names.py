import time
import warnings
from configparser import ConfigParser
from typing import Any

import ArabicNames
import pandas as pd
import structlog
from faker import Faker

from name_matching.config import read_config
from name_matching.log.logging import configure_structlog, configure_tqdm
from name_matching.utils.cli import basic_argparser

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Instantiate configuration class
config = read_config()


class SyntheticNamesGenerator:
    """
    Class for synthetic names generation.
    """

    def __init__(
        self,
        logger: Any,
        config: ConfigParser,
    ) -> None:
        """
        Inits the SyntheticNamesGenerator class.

        :param logger: The logging object
        :param config: The config parser
        """

        # Set the instance variables
        self.logger = logger if logger is not None else structlog.get_logger()
        self.config = config

        # Define the column names
        self.full_name_col = self.config["DATA.COLUMNS"]["FULL_NAME_COL"]
        self.first_name_col = self.config["DATA.COLUMNS"]["FIRST_NAME_COL"]
        self.last_name_col = self.config["DATA.COLUMNS"]["LAST_NAME_COL"]

    def gen_fake_person_names(
        self, faker: Faker, num: int, is_asian: bool = False
    ) -> pd.DataFrame:
        """
        Generate fake person names.

        :param faker: The Faker object
        :param num: Number of names to generate
        :param is_asian: Whether to generate Asian names
        :return: DataFrame with fake person names
        """

        if is_asian:
            first_names = [faker.first_romanized_name() for _ in range(num)]
            last_names = [faker.last_romanized_name() for _ in range(num)]

            # Sample Arabic names
            num_arabic = int(0.25 * num)
            df_first_name = ArabicNames.firstName
            df_first_name = df_first_name.sample(n=num_arabic, ignore_index=True)

            df_last_name = ArabicNames.lastName
            df_last_name = df_last_name.sample(n=num_arabic, ignore_index=True)

            # Add Arabic names to the list
            first_names.extend(df_first_name["Name"].tolist())
            last_names.extend(df_last_name["Name"].tolist())
        else:
            first_names = [faker.first_name() for _ in range(num)]
            last_names = [faker.last_name() for _ in range(num)]

        full_names = [
            first + " " + last for first, last in zip(first_names, last_names)
        ]

        df_person_names = pd.DataFrame(
            {
                self.full_name_col: full_names,
                self.first_name_col: first_names,
                self.last_name_col: last_names,
            }
        )

        return df_person_names

    def gen_fake_orga_names(self, faker: Faker, num: int) -> pd.DataFrame:
        """
        Generate fake organization names.

        :param faker: The Faker object
        :param num: Number of names to generate
        :return: DataFrame with fake organization names
        """

        orga_names = [faker.company() for _ in range(num)]
        first_names = [name for name in orga_names]
        last_names = [name for name in orga_names]
        full_names = [name for name in orga_names]

        df_orga_names = pd.DataFrame(
            {
                self.full_name_col: full_names,
                self.first_name_col: first_names,
                self.last_name_col: last_names,
            }
        )
        return df_orga_names


def main():
    """Run generate names script"""
    parser = basic_argparser()
    parser.description = "Synthetic Name Generation for Name Matching Model"
    parser.allow_abbrev = True
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--n_persons",
        help="Number of person names",
        default=700,
        required=False,
        type=int,
    )
    optional.add_argument(
        "--n_orgas",
        help="Number of organization names",
        default=300,
        required=False,
        type=int,
    )

    args = parser.parse_args()
    configure_structlog(args.silent)
    configure_tqdm(args.disable_tqdm)
    logger = structlog.get_logger()
    logger.info("GENERATE_NAMES_RUN_SCRIPT", **vars(args))

    # Start clocking the time
    start_time = time.time()
    logger.info("SYNTHETIC_NAME_GENERATION_FOR_NAME_MATCHING")

    # Define some column names
    ent_type_col = config["DATA.COLUMNS"]["ENT_TYPE_COL"]

    # Create a synthetic names generator object
    generator = SyntheticNamesGenerator(logger=logger, config=config)

    # Generate synthetic Western names
    num_western = args.n_persons
    western_langs = [
        "en_GB",
        "en_US",
        "fr_FR",
        "fr_CH",
        "de_DE",
        "it_IT",
        "es_ES",
        "pt_PT",
        "nl_NL",
    ]
    faker_western = Faker(western_langs)
    df_person_western = generator.gen_fake_person_names(
        faker=faker_western, num=num_western
    )
    logger.info(
        "WESTERN_PERSON_DF", df="df_western_person", shape=df_person_western.shape
    )

    # Generate synthetic Asian names
    asian_langs = ["zh_CN", "zh_TW", "ja_JP"]
    faker_asian = Faker(asian_langs)
    num_asian = (len(asian_langs) + 1) / len(western_langs) * num_western
    num_asian = int(num_asian)  # +1 above for Arabic names
    df_person_asian = generator.gen_fake_person_names(
        faker=faker_asian, num=num_asian, is_asian=True
    )
    logger.info("ASIAN_PERSON_DF", df="df_asian_person", shape=df_person_asian.shape)

    # Combine both Western and Asian names
    df_person_names = pd.concat([df_person_western, df_person_asian], ignore_index=True)
    logger.info("PERSON_NAMES_DF", df="df_person_names", shape=df_person_names.shape)

    # Generate synthetic organization names
    df_orga_names = generator.gen_fake_orga_names(faker=faker_western, num=num_western)
    logger.info("ORGA_NAMES_DF", df="df_orga_names", shape=df_orga_names.shape)

    # Combine both person and orga names
    df_person_names[ent_type_col] = "PERS"
    df_orga_names[ent_type_col] = "ORGA"
    df_mock_names = pd.concat([df_person_names, df_orga_names], ignore_index=True)
    logger.info("MOCK_NAMES_DF", df="df_mock_names", shape=df_mock_names.shape)

    # Save to disk
    filename_out = config["DATAPATH.RAW"]["MOCK_SAMPLE"]
    df_mock_names.to_csv(filename_out, index=False)
    logger.info("OUTPUT_SAVED_TO", file=filename_out)

    end_time = time.time()
    time_taken = float(end_time - start_time) / 60
    logger.info("TOTAL_RUNTIME", time_taken=round(time_taken, 4), unit="minutes")


if __name__ == "__main__":
    main()

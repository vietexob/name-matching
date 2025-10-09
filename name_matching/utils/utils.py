import re
import random
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from name_matching.config import read_config

# Instantiate configuration class
config = read_config()

# String manipulation libraries
ps = PorterStemmer()

# Regex patterns
STOPWORDS = [word.upper() for word in stopwords.words("english")]
REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")


def process_text_standard(
    text: str,
    remove_numbers: bool = True,
    remove_stopwords: bool = True,
    stem: bool = False,
) -> str:
    """
    Standard procedure to normalize free text.

    :param text: Input free text
    :param remove_numbers: Whether to remove numeric tokens
    :param remove_stopwords: Whether to remove stopwords
    :param stem: Whether to stem the words in text
    :return: Normalized output text
    """

    # Remove non-word symbols
    processed_text = REPLACE_BY_SPACE_RE.sub(" ", text)

    # Remove punctuations
    processed_text = re.sub(r"[^\w\s]", " ", processed_text)

    if remove_numbers:
        processed_text = remove_or_extract_numeric_tokens(
            processed_text, is_removal=True
        )
    else:
        processed_text = " ".join(processed_text.split())

    if remove_stopwords:
        processed_text = " ".join(
            [w for w in processed_text.split() if w not in STOPWORDS]
        )

    if stem:
        # Stem the words
        processed_text = " ".join([ps.stem(w) for w in processed_text.split()])

    return processed_text


def remove_or_extract_numeric_tokens(text: str, is_removal: bool = True) -> str:
    """
    Removes or extracts numeric tokens from the input text.

    :param text: Input text
    :param is_removal: Flag indicating if numeric tokens are removed from text or extracted and returned
    :return: Output string
    """

    tokens = text.split()

    if is_removal:
        new_tokens = [token for token in tokens if not token.isnumeric()]
    else:
        # Numeric tokens are extracted and returned
        new_tokens = [token for token in tokens if token.isnumeric()]

    new_text = " ".join(new_tokens)
    return new_text


def generate_typo_name(name: str, prob_flip: float = 0.25) -> str:
    """
    Generates random typo errors in string name.

    :param name: Input string name
    :param prob_flip: Probability of error per string token
    :return: Output string (with typos)
    """

    assert 0 < prob_flip < 1, "Invalid probability passed!"

    if len(name) == 0:
        return ""

    typo_name = []
    name_tokens = name.split()
    rand_set = string.ascii_uppercase + " "

    for token in name_tokens:
        rand = random.random()
        if rand <= prob_flip:
            idx = random.choice(range(len(token)))
            new_token = "".join(
                [
                    token[i] if i != idx else random.choice(rand_set)
                    for i in range(len(token))
                ]
            )
            typo_name.append(new_token)
        else:
            typo_name.append(token)

    typo_name = " ".join(typo_name)
    return typo_name

import random
import re
import string
from typing import List

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pydantic import BaseModel, Field
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from name_matching.config import read_config

# Instantiate configuration class
config = read_config()

# String manipulation libraries
ps = PorterStemmer()

# Regex patterns
STOPWORDS = [word.upper() for word in stopwords.words("english")]
REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")


# Define the Pydantic model for the response
class AliasesResponse(BaseModel):
    aliases: List[str] = Field(
        description="List of valid aliases for the given name",
        max_length=10,
        min_length=1,
    )


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


def get_mistral_response(
    client=None,
    system_prompt="",
    user_prompt="",
    model="mistral-small-latest",
    messages=[],
):
    """
    Get response from Mistral model.
    """

    assert client is not None, "Mistral client cannot be None!"
    assert model, "Model name cannot be empty!"

    if len(messages) == 0:
        assert len(user_prompt) > 0, "User prompt cannot be empty!"

        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
    else:
        chat_response = client.chat.complete(model=model, messages=messages)

    return chat_response.choices[0].message.content


def generate_aliases(
    client, system_prompt, full_name, first_name="", last_name="", model="gpt-4.1-mini"
):
    """
    Generate aliases for a given name with prompt caching enabled.
    """

    assert client is not None, "OpenAI client cannot be None!"

    # Create the user prompt with the parameters
    if first_name and last_name:
        user_prompt = (
            f"Full name: {full_name}\nFirst name: {first_name}\nLast name: {last_name}"
        )
    else:
        user_prompt = f"Organization name: {full_name}"

    # Make the API call with prompt caching
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},  # Enable caching
                    }
                ],
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format=AliasesResponse,
        temperature=0.7,
        max_tokens=200,
    )

    # Extract the parsed Pydantic object
    aliases_obj = response.choices[0].message.parsed

    # Return as comma-separated string
    return "; ".join(aliases_obj.aliases)


def plot_roc_auc(
    y_test: List[float], y_pred_prob: List[float], filename_out: str = ""
) -> None:
    """
    Plots the ROC curve and computes its AUC.

    :param y_test: List of truth values
    :param y_pred_prob: List of predicted probabilities
    :param filename_out: Output filename (figure)
    """

    assert len(y_test) == len(y_pred_prob), "Input lists are not of the same length!"

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr) * 100

    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}%")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    # plt.show()

    if len(filename_out) > 0:
        # Save figure to disk
        plt.savefig(filename_out, bbox_inches="tight")
    plt.close()


def plot_precision_recall_auc(
    y_test: List[float], y_pred_prob: List[float], filename_out: str = ""
) -> float:
    """
    Plots the Precision-Recall curve and computes its AUC.

    :param y_test: List of truth values
    :param y_pred_prob: List of predicted probabilities
    :param filename_out: Output filename (figure)
    :return: PR-AUC score
    """

    assert len(y_test) == len(y_pred_prob), "Input lists are not of the same length!"

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision) * 100

    plt.title("Precision-Recall Curve")
    plt.plot(recall, precision, "b", label=f"AUC = {pr_auc:.2f}%")
    plt.legend(loc="lower left")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    # plt.show()

    if len(filename_out) > 0:
        # Save figure to disk
        plt.savefig(filename_out, bbox_inches="tight")
    plt.close()

    return pr_auc

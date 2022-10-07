import pandas as pd
from langdetect import detect
from googletrans import Translator
from symspellpy import SymSpell
import pkg_resources
from contractions import contractions_dict
import re
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer

translator = Translator()

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load necessary dictionaries for spelling correction
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

# Load dictionaries as stated in documentation https://symspellpy.readthedocs.io/en/latest/examples/lookup_compound.html
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


def translate_to_en(text):
    """
    Function which uses googletrans library to detect language and translate it if it is not English.
    :param text: Python text to be processed.
    :return: The original text if the detected language is English, and the translated text into English otherwise.
    """
    detected_language = detect(text)
    if detected_language == 'en':
        return text
    else:
        return translator.translate(text, src=detected_language, dest='en').text


def get_avg_word_len(text):
    """
    Function which returns the average word length within a paragraph.
    :param text: The text to search through.
    :return: The average word length from the text.
    """
    words = text.split()
    total_word_len = 0
    for word in words:
        total_word_len += len(word)

    return total_word_len/len(words)


def correct_spelling(text):
    """
    Function which attempts to correct spelling error within a text.
    This is done using SymSpellPy library.
    :param text: The text to correct.
    :return: A corrected version of the text, as understood by the library.
    """
    return sym_spell.lookup_compound(text, max_edit_distance=2)[0].term


def expand_contractions(text):
    """
    Function which expands the words in a text given a static dictionary from contraction python library.
    The assumption is that the text is in lowercase.
    :param text: The text to modify.
    :return:
    """
    if type(text) is str:
        for key in contractions_dict.keys():
            value = contractions_dict[key]
            text = text.replace(key, value)
    return text


def get_tokens(text_series):
    """
    Function which returns tokens from a pandas.Series containing rows of text.
    A token is of form _TOKEN_ or _TOKEN_TOKEN_ or _TOKEN_TOKEN_TOKEN_
    :param text_series: the pandas.Series which contains rows of ticket texts.
    :return: A set of all unique tokens present in all the tickets.
    """
    token_set = set()
    # A regex pattern to match tokens in our text. It is resistant to tokens that are close like
    # _PRODUCT_NAME__PRODUCT_NAME_
    token_regex = r'(_{1,2})([A-Z]+_{0,2}[A-Z]*)\1'
    for index, text in text_series.items():
        regex_results = re.findall(token_regex, text)
        # The findall regex function will return the groups. Therefore we recreate the token by using the found groups.
        tokens_in_text = [group[0] + group[1] + group[0] for group in regex_results]
        for token in tokens_in_text:
            if token not in token_set:
                token_set.add(token)
    return token_set


def get_email_count(text):
    """
    Get the number of emails in the text.
    :param text: The text to search through.
    :return: The number of emails in the text.
    """
    mail_regex = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'
    return len(re.findall(mail_regex, text))


def remove_emails(text):
    """
    Remove the emails from text.
    :param text: The text to modify.
    :return: The text without any email that was initially in.
    """
    mail_regex = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

    return re.sub(mail_regex, '', text)


def get_url_count(text):
    """
    Get the number of URLs in the text.
    :param text: The text to search through.
    :return: The number of URLs in the text.
    """
    url_regex = r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?'

    return len(re.findall(url_regex, text))


def remove_urls(text):
    """
    Remove the URLs from text.
    :param text: The text to modify.
    :return: The text without any URL that was initially in.
    """
    url_regex = r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?'

    return re.sub(url_regex, '', text)


def remove_special_characters(text):
    """
    Function which removes all the special characters, besides - and _ from text.
    :param text: The text to modify.
    :return: Text without any special characters besides - and _. Special characters are replaced by space.
    """
    special_chars_regex = r'[^A-Za-z0-9 \-\_]+'
    return re.sub(special_chars_regex, ' ', text)


def remove_single_characters(text):
    """
    Function which removes all the single characters from text.
    :param text: The text to modify.
    :return: Text containing words with a length >= 2.
    """
    return " ".join([word for word in text.split() if len(word) > 1])


def preprocess_raw_text(text, token_dictionary, stemmer):
    """
    Function that combines all of the processing functions to clean up raw text. It does the following things:

    1. Measure text length and add it to manual_features list at index 0.
    2. Measure number of words and add it to manual_features list at index 1.
    3. Measure average word length and add it to manual_features list at index 2.
    4. Use function correct_spelling to try to correct the errors in the text.
    5. Use function expand_contractions to expand word like won't and don't to will not and do not.
    6. Measure the numeric counts of each message (the count of number values present in each message)
    and add it to manual_features at index 3.
    7. Measure the number of each token from token_dictionary found in text and add these values to manual_features
    list; remove them from text after counting to avoid counting token multiple times
    (i.e. a token that can fit inside other token)
    8. Convert text to lowercase.
    9. I will count number of emails in each message and remove them.
    10. I will count and remove any url from the text.
    11. I will count and remove the number of stopwords from the text.
    12. I will remove single characters and special characters from text.
    13. I will stem the remaining words.
    :param stemmer: Stemmer to stem each word of the text.
    :param token_dictionary: dictionary of tokens, where the key is the actual token value and the value is the group
    that the token is part of.
    :param text: The text that will be preprocessed and from which we extract the manual features.
    :return: The preprocessed text, ready to be used as fodder for tf, tf-idf or word embedding techniques.
    """
    manual_features = dict()

    text_length = len(text)
    manual_features['text_length'] = text_length

    number_of_words = len(text.split())
    manual_features["number_of_words"] = number_of_words

    average_word_length = get_avg_word_len(text)
    manual_features["average_word_length"] = average_word_length

    text = expand_contractions(text)

    numeric_counts = len([word for word in text.split() if word.isnumeric()])
    manual_features["numeric_counts"] = numeric_counts

    # Because certain tokens like __PRODUCT_NAME__ contain other tokens (i.e. _NAME_), we would like to count and
    # replace the bigger tokens first. For this we sort the token list based on word length in descending order.
    for token in sorted(token_dictionary.keys(), key=len, reverse=True):
        # If group is not already in manual_features dictionary as a key, initialize value at the group key with 0.
        manual_features[token_dictionary[token]] = manual_features.get(token_dictionary[token], 0) + text.count(token)
        text = text.replace(token, " ")

    text = text.lower()

    # Since email_count is already a feature extracted using the tokens, I will just increase it
    manual_features["email_count"] += get_email_count(text)
    text = remove_emails(text)

    url_count = get_url_count(text)
    manual_features["url_count"] = url_count
    text = remove_urls(text)

    stop_word_count = len([word for word in text.split() if word in STOP_WORDS])
    manual_features["stop_word_count"] = stop_word_count
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])

    text = remove_special_characters(text)
    text = remove_single_characters(text)

    text = " ".join([stemmer.stem(word) for word in text.split()])

    return text, manual_features


def preprocess_text_series(text_series, token_dictionary, with_manual_features=True):
    """
    Function which takes a pandas.Series comprised of texts and applies raw text preprocessing on it.
    If the with_manual_features flag is set, it also adds the manual features extracted to the resulting dataframe
    :param token_dictionary: dictionary of tokens, where the key is the general meaning of the token and what we will
    use for column naming and the value is a list of actual values found in the text.
    :param text_series: A pandas.Series containing rows of text.
    :param with_manual_features: Flag. If set to False, do not append manual features to the resulting DataFrame.
    Append them if set to True. Default is True.
    :return: pd.DataFrame containing the preprocessed text and the 22 manual features extracted from it.
    """
    stemmer = PorterStemmer()

    # Keep index values to ensure same order for messages and manual features and labels
    index_values = []
    text_values = []

    if with_manual_features:
        manual_features_list = []

    for index, value in text_series.items():
        processed_text, manual_features = preprocess_raw_text(value, token_dictionary, stemmer)
        index_values.append(index)
        text_values.append(processed_text)
        if with_manual_features:
            manual_features_list.append(manual_features)

    text_df = pd.DataFrame(text_values, index=index_values, columns=["text"])

    if with_manual_features:
        manual_features_df = pd.DataFrame(manual_features_list, index=index_values)
        text_df = pd.concat([text_df, manual_features_df], axis=1)

    return text_df


























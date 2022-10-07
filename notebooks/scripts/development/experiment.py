from googletrans import Translator
import pandas as pd

translator = Translator()


def enhance_text(text, languages=None):
    """
    Function that takes a string, translates it into all of the languages passed,
    back to English and returns them as a list.
    :param text: The original text.
    :param languages: The languages in which to translate the text.
    :return: A list of 6 English written texts.
    """
    if languages is None:
        languages = ['de', 'ro', 'nl', 'fr', 'fi', 'hu']
    enhanced_texts = []
    for language in languages:
        translated = translator.translate(text, dest=language, src='en').text
        translated_back = translator.translate(translated, dest='en', src=language).text
        enhanced_texts.append(translated_back)

    return enhanced_texts


def enhance_series(text_series, languages=None):
    """
    Function which takes a pandas.Series comprised of texts and enhances them len(languages) times.
    :param text_series: A pandas.Series representing the data to enhance.
    :param languages: The languages to use when translating into the non-English language. Default is
    ['de', 'ro', 'nl', 'fr', 'fi', 'hu'] (i.e. increase data size by a factor of 6)
    :return: A pandas.Series containing artificially enhanced data by a factor of len(languages)
    """
    enhanced_list = []
    for text in text_series:
        enhanced_list.extend(enhance_text(text, languages))

    return pd.Series(enhanced_list)

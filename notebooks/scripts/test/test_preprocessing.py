from unittest import TestCase
from preprocessing import get_avg_word_len, correct_spelling, translate_to_en, expand_contractions, \
    get_tokens, get_email_count, remove_emails, get_url_count, remove_urls, remove_single_characters, \
    remove_special_characters, preprocess_raw_text
import pandas as pd
from nltk.stem import PorterStemmer
import json

TOKEN_DICTIONARY = {'_ADDRESS_': 'address_count',
                    '__ADDRESS__': 'address_count',
                    '_LOCATION_': 'address_count',
                    '__PLACE__': 'address_count',
                    '_COMPANNY_': 'company_name_count',
                    '_COMPANY_': 'company_name_count',
                    '__COMPANY_NAME__': 'company_name_count',
                    '__COMPANY__': 'company_name_count',
                    '_CREDIT_': 'credit_card_count',
                    '__CREDIT_CARD__': 'credit_card_count',
                    '_DATE_': 'date_count',
                    '_MONTH_': 'date_count',
                    '__DATE__': 'date_count',
                    '_INVOICE_NUMBER_': 'invoice_count',
                    '__INVOICE_NUMBER__': 'invoice_count',
                    '_ITEM_PHOTO_': 'photo_count',
                    '_NAME_': 'name_count',
                    '__NAMES__': 'name_count',
                    '__NAME__': 'name_count',
                    '_ORDER_NUMBBER_': 'order_count',
                    '_ORDER_NUMBER_': 'order_count',
                    '__ORDER_NUMBER__': 'order_count',
                    '_OTHER_PI_': 'other_pi_count',
                    '__OTHER_PI__': 'other_pi_count',
                    '_PHONE_': 'phone_count',
                    '__PHONE__': 'phone_count',
                    '_PRICE_': 'price_count',
                    '__AMOUNT__': 'price_count',
                    '_PRODUCT_': 'product_count',
                    '_PRODUCT_NAME_': 'product_count',
                    '__PRODUCTS_NAMES__': 'product_count',
                    '__PRODUCT_NAMES__': 'product_count',
                    '__PRODUCT_NAME__': 'product_count',
                    '__PRODUCT__NAMES__': 'product_count',
                    '__PRODUCT__NAME__': 'product_count',
                    '_TRACKING_NUMBER_': 'tracking_number_count',
                    '__TRACKING_NUMBER__': 'tracking_number_count',
                    '_URL_': 'url_count',
                    '__URL__': 'url_count',
                    '__DISCOUNT_CODE__': 'discount_code_count',
                    '__EMAIL__': 'email_count',
                    '__REFERENCE_NUMBER__': 'reference_number_count'}


class TestPreprocessing(TestCase):
    def test_translate_to_en(self):
        # Given
        texts = ["It is a wonderful night let's play", "Hallo, ich möchte wissen, wo es ist"]

        # When
        translated_texts = [translate_to_en(x) for x in texts]

        # Then
        self.assertEqual(texts[0], translated_texts[0])
        self.assertNotEqual(texts[1], translated_texts[1])

    def test_get_avg_word_len(self):
        # Given
        texts = ["It is a wonderful night let's play", "Wow    NICE"]

        # When
        counts = [get_avg_word_len(x) for x in texts]

        # Then
        self.assertListEqual(counts, [4, 3.5])

    # A simple test to make sure some common mistakes are corrected
    # A spell checker will never get a text 100% correctly
    def test_correct_spelling(self):
        # Given
        text = "reciept notdrive cleer hopin borke inveat"

        # When
        corrected_text = correct_spelling(text)

        # Then
        self.assertEqual("receipt not drive clear hop in broke invest", corrected_text)

    def test_expand_contractions(self):
        # Given
        text = "I'll be right back with you I can't wait to see you. I shan't forget and won't"

        # When
        expansion = expand_contractions(text)

        # Then
        self.assertEqual(expansion, "I will be right back with you I cannot wait to see you. "
                                    "I shall not forget and will not")

    def test_get_tokens(self):
        # Given
        tickets_series = pd.Series(
            ["My order number is __ORDER_NUMBER__ , and __COMPANY__kept I contacted _COMPANY_and gave",
             "Ahola __PRODUCT_NAMES__ hats Regards __NAME__",
             "- __EMAIL__ Contact me by email Portuguese._DATE_I bought women _PRODUCT_NAME_ In",
             "return my _PRODUCT_NAME__PRODUCT_NAME_  as it",
             "_DATE__PRODUCT_NAME_ __PRODUCT__NAME__"])

        # When
        tokens = get_tokens(tickets_series)

        # Then
        self.assertSetEqual(tokens, {"__ORDER_NUMBER__", "__COMPANY__", "_COMPANY_", "__PRODUCT_NAMES__",
                                     "__NAME__", "__EMAIL__", "_DATE_", "_PRODUCT_NAME_", "__PRODUCT__NAME__",
                                     "_DATE__PRODUCT_"})
        # You can see that one thing we cannot account for is mistakes like _DATE__PRODUCT_NAME_.
        # We will have to look through the tokens manually

    def test_get_email_count(self):
        # Given
        text = "I wanna check with jo@bee.com and michael@wawawawa.brough and perhaps jimmy-1212@me.com and " \
               "?+-231aa@ha.do wahoo@me?com"

        # When
        count = get_email_count(text)

        # Then
        self.assertEqual(4, count)

    def test_remove_emails(self):
        # Given
        text = "I wanna check with jo@bee.com and michael@wawawawa.brough and perhaps jimmy-1212@me.com and " \
               "?+-231aa@ha.do wahoo@me?com"

        # When
        new_text = remove_emails(text)

        # Then
        self.assertEqual("I wanna check with  and  and perhaps  and ? wahoo@me?com", new_text)

    def test_get_url_count(self):
        # Given
        text = "@switchfoot http://twitpic.com/2y1zl - awww, t... httv://www.me.com ftp://ro.know.me/"

        # When
        count = get_url_count(text)

        # Then
        self.assertEqual(2, count)

    def test_remove_urls(self):
        # Given
        text = "@switchfoot http://twitpic.com/2y1zl - awww, t... httv://www.me.com ftp://ro.know.me/"

        # When
        new_text = remove_urls(text)

        # Then
        self.assertEqual("@switchfoot  - awww, t... httv://www.me.com ", new_text)

    def test_remove_special_characters(self):
        # Given
        text = "So cool dude ... hope u can be.so.nice and you+me=great h@h@h@ see ya @@@@ wanna-be laughing_face"

        # When
        new_text = remove_special_characters(text)

        # Then
        self.assertEqual("So cool dude   hope u can be so nice and you me great h h h  see ya   wanna-be laughing_face",
                         new_text)

    def test_remove_single_characters(self):
        # Given
        text = "So cool dude   hope u can be so nice and you me great h h h  see ya   wanna-be laughing_face"

        # When
        new_text = remove_single_characters(text)

        # Then
        self.assertEqual("So cool dude hope can be so nice and you me great see ya wanna-be laughing_face", new_text)

    # Simply used to ensure that the big function works properly by looking at it with the debugger.
    def test_preprocess_raw_text(self):
        # Given
        stemmer = PorterStemmer()
        texts = ["Hi \nRe order _ORDER_NUMBER_\nMy daughter needs to return part of this order please _ORDER_NUMBER_ "
                 "can you provide return details.",
                 "Thank you for your time  - cbdelo.us@gmail.com"]

        # When
        processed_texts = [preprocess_raw_text(x, TOKEN_DICTIONARY, stemmer) for x in texts]

        # Then
        self.assertEqual('hi order daughter need return order provid return detail', processed_texts[0][0])
        self.assertEqual("thank time", processed_texts[1][0])

        self.assertEqual(2, processed_texts[0][1]["order_count"])
        self.assertEqual(9, processed_texts[0][1]["stop_word_count"])
        self.assertEqual(0, processed_texts[0][1]["email_count"])
        self.assertEqual(0, processed_texts[1][1]["order_count"])
        self.assertEqual(1, processed_texts[1][1]["email_count"])
        self.assertEqual(3, processed_texts[1][1]["stop_word_count"])

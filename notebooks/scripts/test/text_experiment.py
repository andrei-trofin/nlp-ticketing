import unittest
from experiment import enhance_text


class MyTestCase(unittest.TestCase):
    def test_enhance_text(self):
        # Given
        original_text = "I ordered a pair of __PRODUCTS_NAMES__ shoes, but I never received a confirmation email. " \
                        "Makes me think I didn't complete the purchase process somehow?? Can you confirm that" \
                        " I placed an order? __NAME__ - __EMAIL__"

        # When
        enhanced_texts = enhance_text(original_text)

        # Then
        for text in enhanced_texts:
            self.assertNotEqual(original_text, text)
        # print('\n'.join(enhanced_texts))


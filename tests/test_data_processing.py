import unittest
import pandas as pd
import numpy as np
from src.data.preprocessing import clean_text, preprocess_text, standardize_categories


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.sample_text = "This is a SAMPLE text with XXXX and numbers 123!"
        self.sample_df = pd.DataFrame({
            'product': ['Credit Cards', 'credit card', 'CREDIT-CARD'],
            'company': ['WELLS FARGO BANK', 'WellsFargo', 'wells fargo bank'],
            'date_received': ['2023-01-01', '2023-01-02', '2023-01-03']
        })

    def test_clean_text(self):
        """Test text cleaning function"""
        cleaned = clean_text(self.sample_text)
        self.assertEqual(
            cleaned, "this is a sample text with [REDACTED] and numbers")

        # Test handling of None/NaN
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(np.nan), "")

    def test_preprocess_text(self):
        """Test text preprocessing function"""
        tokens = preprocess_text(self.sample_text)

        # Check if stopwords are removed
        self.assertNotIn('is', tokens)
        self.assertNotIn('a', tokens)

        # Check if short tokens are removed
        self.assertTrue(all(len(token) > 2 for token in tokens))

        # Check handling of None/NaN
        self.assertEqual(preprocess_text(None), [])
        self.assertEqual(preprocess_text(np.nan), [])

    def test_standardize_categories(self):
        """Test category standardization"""
        # Test product standardization
        standardized = standardize_categories(self.sample_df, 'product')
        self.assertTrue(all(standardized == 'credit card'))

        # Test company standardization
        standardized = standardize_categories(self.sample_df, 'company')
        self.assertTrue(all(standardized == 'wells fargo'))


if __name__ == '__main__':
    unittest.main()

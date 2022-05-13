import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import pandas as pd
from scripts.eda import EDA


class TestCases(unittest.TestCase):

    def __init__(self):
        self.df = pd.read_csv("../data/data.csv")

    def test_generate_pipeline(self):
        """
        Test that eda generates a pipeline
        """
        analyzer = EDA(self.df)
        numeric_pipeline = analyzer.generate_pipeline("numeric")
        self.assertTrue(numeric_pipeline)
    
    
if __name__ == '__main__':
    unittest.main()

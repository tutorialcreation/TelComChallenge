import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import pandas as pd
from scripts.eda import EDA


class TestCases(unittest.TestCase):
    
    df = pd.read_csv("data/data.csv")
    analyzer = EDA(df)
    numeric_pipeline = analyzer.generate_pipeline("numeric")
    numeric_transformation =  analyzer.generate_transformation(numeric_pipeline,"numeric","number")
    numerical_features = analyzer.store_features("numeric","number")
    
    def test_generate_pipeline(self):
        """
        Test that eda generates a pipeline
        """
        self.assertTrue(self.numeric_pipeline)
    
    
    def test_store_features(self):
        """
        - testing store features
        """
        
        self.assertTrue(self.numerical_features)

    def test_generate_transformations(self):
        """
        - testing generating transforms
        """
        self.assertTrue(self.numeric_transformation.any())
    
    def test_frame_transforms(self):
        """
        - testing frame_transforms
        """
        test = True
        numeric_df = self.analyzer.frame_transforms(self.numeric_transformation,self.numerical_features)
        if numeric_df.empty:
            test = False
        self.assertTrue(test)

    def test_top_x_column(self):
        """
        - testing top_x_column
        """
        values = self.analyzer.top_x_column(10,"Handset Manufacturer","purple")

        self.assertTrue(values)

    def test_top_x_by_y_cols(self):
        """
        - testing store features
        """
        values_ = self.analyzer.top_x_by_y_cols('Handset Manufacturer','Handset Type',3,5)
    
        self.assertTrue(values_)

    def test_aggregation_cols(self):
        """
        - testing aggregation_cols
        """
        test=True
        aggregations = self.analyzer.aggregation_cols('MSISDN/Number','Total UL (Bytes)',True)
        if aggregations.empty:
            test = False
        self.assertTrue(test)

    def test_non_graphical_analysis(self):
        """
        - testing non_graphical_analysis
        """
        test = True
        analysis_1 = self.analyzer.non_graphical_analysis(self.numerical_features,"univariate",3)
        if analysis_1.empty:
            test = False
        self.assertTrue(test)

        
    def test_pca_analysis(self):
        """
        - testing pca_analysis
        """
        test = True
        analysis_4 = self.analyzer.pca_analysis(self.numerical_features,"numeric",5,49,1)
        if analysis_4.empty:
            test = False
        self.assertTrue(test)

    
    
if __name__ == '__main__':
    unittest.main()

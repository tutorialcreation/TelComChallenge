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
    
    # values = analyzer.top_x_column(10,"Handset Manufacturer","purple")
    # fig,values = analyzer.top_x_column(10,"Handset Manufacturer","purple",online=True)
    # values_ = analyzer.top_x_by_y_cols('Handset Manufacturer','Handset Type',3,5)
    # aggregations = analyzer.aggregation_cols('MSISDN/Number','Total UL (Bytes)',True)
    # analysis_1 = analyzer.non_graphical_analysis(numerical_features,"univariate",3)
    # analysis_2 = analyzer.graphical_analysis(numerical_features,"univariate","curve",x=1)
    # analysis_3 = analyzer.non_graphical_analysis(numerical_features,"multivariate",1,4)
    # analysis_4 = analyzer.pca_analysis(numerical_features,"numeric",10,49,1)
    # analysis_5 = analyzer.categorize_based_on_deciles(numerical_features,49)

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

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    # def test_store_features(self):
    #     """
    #     - testing store features
    #     """
    #     storage = self.analyzer.store_features("numeric","number")
    #     self.assertTrue(storage)

    
    
if __name__ == '__main__':
    unittest.main()

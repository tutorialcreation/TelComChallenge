import streamlit as st
import pandas as pd
from scripts.eda import EDA

st.title("Telecommunication User Analytics")

df = pd.read_csv("../data/telcom.csv")
analyzer = EDA(df)
numeric_pipeline = analyzer.generate_pipeline("numeric")
numerical_features = analyzer.store_features("numeric","number")
numeric_transformation = analyzer.generate_transformation(numeric_pipeline,"numeric","number")
numeric_df = analyzer.frame_transforms(numeric_transformation,numerical_features)
values = analyzer.top_x_column(10,"Handset Manufacturer","purple")
values_ = analyzer.top_x_by_y_cols('Handset Manufacturer','Handset Type',3,5)
aggregations = analyzer.aggregation_cols('MSISDN/Number','Total UL (Bytes)',True)
analysis_1 = analyzer.non_graphical_analysis(numerical_features,"univariate",3)
analysis_2 = analyzer.graphical_analysis(numerical_features,"univariate","curve",x=1)
analysis_3 = analyzer.non_graphical_analysis(numerical_features,"multivariate",1,4)
analysis_4 = analyzer.pca_analysis(numerical_features,"numeric",10,49,1)
analysis_5 = analyzer.categorize_based_on_deciles(numerical_features,49)
indexer = analyzer.map_index_to_feature(2,numerical_features)
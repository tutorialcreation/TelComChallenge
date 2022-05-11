import streamlit as st
import pandas as pd
from scripts.eda import EDA

st.title("Telecommunication User Analytics")
st.sidebar.title("Configurations")

df = pd.read_csv("data/telcom.csv")
analyzer = EDA(df)
numeric_pipeline = analyzer.generate_pipeline("numeric")
numerical_features = analyzer.store_features("numeric","number")
categorical_features = analyzer.store_features("categorical","number")
numeric_transformation = analyzer.generate_transformation(numeric_pipeline,"numeric","number")
numeric_df = analyzer.frame_transforms(numeric_transformation,numerical_features)
# measures of location
st.sidebar.subheader("Measures of location")
numeric_variable= st.sidebar.selectbox('Numeric Variables (x1):',numerical_features)
numeric_variable_ = analyzer.map_index_to_feature(numeric_variable,numerical_features)
numeric_variable_1= st.sidebar.selectbox('Numeric Variables (x2):',numerical_features)
numeric_variable_1_ = analyzer.map_index_to_feature(numeric_variable_1,numerical_features)
categorical_variable = st.sidebar.selectbox('Categorical Variables (x1):',categorical_features)
categorical_variable_ = analyzer.map_index_to_feature(categorical_variable,categorical_features)
categorical_variable_1 = st.sidebar.selectbox('Categorical Variables (x2):',categorical_features)
categorical_variable_1_ = analyzer.map_index_to_feature(categorical_variable_1,categorical_features)
top_x = int(st.sidebar.text_input("top x",3))
top_y = int(st.sidebar.text_input("top y",5))

if top_x and categorical_variable_:
    fig,values = analyzer.top_x_column(top_x,categorical_variable,"purple",online=True)
    st.pyplot(fig)

if top_x and categorical_variable_ and top_y:
    values = analyzer.top_x_by_y_cols(categorical_variable,categorical_variable_1,top_x,top_y)
    for i in values:
        st.dataframe([i])

st.sidebar.subheader("Measures of Central Tendency")
if st.sidebar.checkbox("aggregate: min,max,mean based on two variables"):
    if numeric_variable and numeric_variable_1:
        st.subheader("Measures of Central Tendency")
        aggregations = analyzer.aggregation_cols(numeric_variable,numeric_variable_1,True)
        st.dataframe(aggregations)

type_ = st.sidebar.radio("What type of analysis will you undertake?",
("univariate","bivariate","multivariate"))

if type_ and numeric_variable_:
    if type_ == "univariate":
        analysis_type_1 = analyzer.non_graphical_analysis(numerical_features,type_,numeric_variable_)
    elif type_ == "bivariate":
        analysis_type_1 = analyzer.non_graphical_analysis(numerical_features,type_,numeric_variable_, numeric_variable_1_)
    elif type_ == "multivariate":
        analysis_type_1 = analyzer.non_graphical_analysis(numerical_features,type_,numeric_variable_, numeric_variable_1_)
    
    st.write(analysis_type_1)

st.sidebar.subheader("PCA Analysis")
components = int(st.sidebar.text_input("no. of components",10))
component_return = int(st.sidebar.text_input("return which component",1))
if components:
    try:
        analysis_type_2 = analyzer.pca_analysis(numerical_features,"numeric",components,numeric_variable_,component_return)
        st.subheader("PCA Analysis")
        st.write(analysis_type_2)
    except Exception as e:
        st.error(e)


st.sidebar.subheader("Measures of dispersion")

if st.sidebar.checkbox("find deciles"):
    try:
        analysis_type_3 = analyzer.categorize_based_on_deciles(numerical_features,numeric_variable_)
        st.write(analysis_type_3)
    except Exception as e:
        st.error(e)

        
analysis_2 = analyzer.graphical_analysis(numerical_features,"univariate","curve",x=1)

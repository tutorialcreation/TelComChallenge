import streamlit as st
import pandas as pd
from models.db import DBOps
from scripts.mlscript import mlscript
import warnings
from xml.etree.ElementInclude import include
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from statistics import mean
import numpy as np
import statsmodels.api as sm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

import matplotlib
plt.style.use('ggplot')

matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import seaborn as sns


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Telecommunication User Analytics")
st.sidebar.title("Configurations")

df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vT_rh0uXzokNtBHxDlXiSOCFSfCIa_TD8I7hNPUA4MUAFcxvk0oknFuyYKRWlC0IR26u59VMMWrThvn/pub?output=csv")
analyzer = mlscript(df)
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
        st.subheader("Measures of dispersion")
        st.write(analysis_type_3)
    except Exception as e:
        st.error(e)


st.sidebar.subheader("Charts")
option = st.sidebar.radio("What type of chart would you like to see?",
("box","hist","curve","scatter"))
if option:
    try:
        analysis_type_3 = analyzer.graphical_analysis(numerical_features,type_,option,x=int(numeric_variable_))
        st.pyplot(analysis_type_3)
    except Exception as e:
        st.error(e)


st.sidebar.subheader("Modeling")
app_df = pd.DataFrame({'customer':df['MSISDN/Number'],
                      'sessions_frequency':df['Bearer Id'],
                      'duration':df['Dur. (ms)']})
app_df['social_media_data'] = df['Social Media DL (Bytes)'] + df['Social Media UL (Bytes)']
app_df['google_data'] = df['Google DL (Bytes)'] + df['Google UL (Bytes)'] 
app_df['email_data'] = df['Email DL (Bytes)'] + df['Email UL (Bytes)'] 
app_df['youtube_data'] = df['Youtube DL (Bytes)'] + df['Youtube UL (Bytes)']
app_df['netflix_data'] = df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']
app_df['gaming_data'] = df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)'] 
app_df['other_data'] = df['Other DL (Bytes)'] + df['Other UL (Bytes)']
app_df['total_data'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']

top_x_duration = int(st.sidebar.text_input("find top x customers based on duration",10))

# if top_x_duration:
#     duration_aggregation = analyzer.aggregation_cols(app_df,'customer','duration')
#     top_customers_duration = duration_aggregation.sort_values(by='duration_max', ascending=False)
#     st.write(top_customers_duration)

st.sidebar.text("Cluster data based on durations, sesssions, and ")
df_to_transform = app_df[app_df.columns.to_list()[1:]]
_,df_to_transform = analyzer.handle_missing_values_numeric(df_to_transform,df_to_transform.columns)
pca = PCA(2)


#Transform the data
df_ = pca.fit_transform(df_to_transform)
 
no_clusters = int(st.sidebar.text_input("Place the number of clusters",2))


kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=42)
y_pred = kmeans.fit_predict(df_)
app_df['y_pred'] = y_pred
labels_ = np.unique(y_pred)
 
#plotting the results:
 
for i in labels_:
    plt.scatter(df_[y_pred == i , 0] , df_[y_pred == i , 1] , label = i)
plt.legend()
plt.show()





st.sidebar.subheader("Satisfaction Analysis")
top_x_satisfied = int(st.sidebar.text_input("Top x most satisfied customers",10))
if top_x_satisfied:
    db = DBOps(is_online=True)
    satisfaction = pd.read_sql('SELECT * FROM userData', db.get_engine())
    x_satisfied = satisfaction.sort_values(by="satisfaction_score",ascending=False).head(top_x_satisfied)
    st.subheader(f"top {top_x_satisfied}  most satisfied customers")
    st.dataframe(x_satisfied)

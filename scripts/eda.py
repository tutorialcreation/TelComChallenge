#!/usr/bin/env python
# coding: utf-8

# ## Telecom User Data Analysis

# ### Libraries


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from statistics import mean
from pandas_profiling import ProfileReport
import numpy as np
import json
import datetime
import math
import statsmodels.api as sm

from datetime import timedelta, datetime

import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import seaborn as sns


class EDA:

    """
    This program/script performs the following
    - eda analysis of the data
    """

    def __init__(self,df):
        """
        purpose:
        -initialize the class
        
        """
        self.df = df


    def generate_pipeline(self,type_="numeric",x=1):
        pipeline = None
        if type_ == "numeric":
            pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', MinMaxScaler())
            ])
        elif type_ == "categorical":
            pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        else:
            pipeline = np.zeros(x)
        return pipeline



    
    def store_features(self, df,type_,value):
        features = [None]
        if type_ == "numeric":
            features = df.select_dtypes(include=value).columns.tolist()
        elif type_ == "categorical":
            features = df.select_dtypes(exclude=value).columns.tolist()
        return features


    
    def generate_transformation(self,pipeline,df,type_,value):
        transformation = None
        if type_=="numeric":
            transformation=pipeline.fit_transform(df.select_dtypes(include=value))
        elif type_ == "categorical":
            transformation=pipeline.fit_transform(df.select_dtypes(exclude=value))
        return transformation


    

    def frame_transforms(self,transform,features):
        return pd.DataFrame(transform,columns=features)


    

    def split_data(self,df,response_variable,split_ratio,get):
        X = df.drop(response_variable, axis=1)
        y = df[response_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, 
                                                        random_state=1121218)
        if get == "X_train":
            return X_train
        elif get == "X_test":
            return X_test
        elif get == "y_train":
            return y_train
        else:
            return y_test
        
        


    

    def handle_missing_values(self,df,x):
        """
        this algorithm does the following
        - remove columns with x percentage of missing values
        - fill the missing values with the mean
        returns:
            - df
            - percentage of missing values
        """
        missing_percentage = round((df.isnull().sum().sum()/\
                reduce(lambda x, y: x*y, df.shape))*100,2)
        null_cols = df.isnull().sum().to_dict()
        for key,val in null_cols.items():
            if val/df.shape[0] > x:
                df.drop([key], axis=1)
            elif  val/df.shape[0] < x and val > 0 and df[key].dtype.kind in 'biufc':
                df.fillna(df[key].mean().round(1), inplace=True)
        return missing_percentage, df


   
    def top_x_column(self,df, x, column,color,online=False):
        handsets_df = pd.DataFrame(columns = [column])
        handsets_df['type'] = df[column].to_list()
        handsets = handsets_df['type'].value_counts()
        fig,ax = plt.subplots()
        ax.tick_params(axis='x',labelsize=10)
        ax.tick_params(axis='y',labelsize=10)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"The {x} Most Frequent {column}")
        handsets[:x].plot(ax=ax,kind='bar',color=color)
        handset_counts = handsets.to_dict()
        top_x = list(handset_counts.keys())
        if online:
            return fig,top_x[:x]
        else:
            return top_x[:x]
        


    
    # Identify the top 5 handsets per top 3 handset manufacturer
    def top_x_by_y_cols(self,df,col_1,col_2,x,y):
        result_df = []
        by_manufacture = df.groupby(col_1,sort=True)
        values = top_x_column(df,x,col_1,"purple")


        for manufacturer, frame in by_manufacture:
            if manufacturer in values:
                result_df.append(frame.sort_values(by=[col_2], ascending=True)[col_2].head(5))
        return result_df
        


    
    def aggregation_cols(self,df,col_1,col_2,trim=False):
        
        grouped = df.groupby(col_1).agg({col_2: [min, max, mean]}) 
        grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
        if trim:
            return grouped.describe()
        return grouped


    
    def non_graphical_analysis(self,df,features,type_,opt,x_=1,y_=1):
        result = None
        if type_ == "univariate":
            for i,key in enumerate(features):
                if i == x:
                    result = pd.DataFrame(df[key].describe())
        elif type_ == "bivariate":
            for i,key in enumerate(features):
                if i == x_:
                    if opt=="regression":
                        y = df[features[y_]]
                        x = df[[key]]
                        x = sm.add_constant(x)
                        model = sm.OLS(y, x).fit()
                        result =  model.summary()
                    elif opt=="corr":
                        result = pd.DataFrame(df[[key,features[y]]].corr())
        elif type_ == "multivariate":
            result = pd.DataFrame(df[features].corr())
        return result
            


  
    # Conduct a Graphical Univariate Analysis by identifying the most suitable plotting options 
    # for each variable and interpret your findings.

    def graphical_analysis(self,df,features,type_,opt,x=1,y=1):
        result = None
        if type_ == "univariate":
            for i,key in enumerate(features):
                if i == x:
                    if opt == 'box':
                        return df.boxplot(column=[key], grid=False, color='black')
                    elif opt == 'hist':
                        return df.hist(column=[key], grid=False, edgecolor='black')
                    elif opt == 'curve':
                        return sns.kdeplot(df[key])
        if type_ == "bivariate":
            for i,key in enumerate(features):
                if i == x:
                    if opt == "scatter":
                        plt.scatter(df[features[x]], df[features[y]])
                        plt.title(f'{features[x]} vs {features[y]}')
                        plt.xlabel(f'{features[x]}')
                        plt.ylabel(f'{features[y]}')
                    
        


    
    # pca analysis
    def setup_pca(self,data,n):
        pca = PCA(n)
        x_ = pca.fit_transform(data)
        return x_, pca




    def pca_analysis(self,df,features,no,x_,component):
        for i,key in enumerate(features):
            if i==x_:
                train = generate_transformation(numeric_pipeline,
                                            pd.DataFrame(split_data(df,key,0.3,"X_train")),
                                            "numeric","number")
                test = generate_transformation(numeric_pipeline,
                                                pd.DataFrame(split_data(df,key,0.3,"X_test")),
                                                "numeric","number")
                pca_train_results, pca_train = setup_pca(train, no)
                pca_test_results, pca_test = setup_pca(test, no)
                names_pcas = [f"PCA Component {i}" for i in range(1, 11, 1)]
                scree = pd.DataFrame(list(zip(names_pcas, pca_train.explained_variance_ratio_)), columns=["Component", "Explained Variance Ratio"])
                df = pd.DataFrame({'PCA':pca_train.components_[component], 'Variable Names':numerical_features})            
                df = df.sort_values('PCA', ascending=False)
                df2 = pd.DataFrame(df)
                df2['PCA']=df2['PCA'].apply(np.absolute)
                df2 = df2.sort_values('PCA', ascending=False)
                return df2
        return

    def categorize_based_on_deciles(self,df,features,x_):
        for i,key in enumerate(features):
            if i==x_:
                df['decile_rank'] = pd.qcut(df[key], 10,labels = False)
                return df.groupby(['decile_rank']).sum()
        return


   
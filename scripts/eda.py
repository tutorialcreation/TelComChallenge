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
        input:
            - df
        returns:
            -df
        """
        self.df = df


    def generate_pipeline(self,type_="numeric",x=1):
        """
        purpose:
            - generate_pipelines for the data
        input:
            - string and int
        returns:
            - pipeline
        """
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



    
    def store_features(self,type_,value):
        """
        purpose:
            - stores features for the data set
        input:
            - string,int,dataframe
        returns:
            - dataframe
        """
        features = [None]
        if type_ == "numeric":
            features = self.df.select_dtypes(include=value).columns.tolist()
        elif type_ == "categorical":
            features = self.df.select_dtypes(exclude=value).columns.tolist()
        return features


    
    def generate_transformation(self,pipeline,type_,value):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        transformation = None
        if type_=="numeric":
            transformation=pipeline.fit_transform(self.df.select_dtypes(include=value))
        elif type_ == "categorical":
            transformation=pipeline.fit_transform(self.df.select_dtypes(exclude=value))
        return transformation


    

    def frame_transforms(self,transform,features):
        """
        purpose:
            - merges the transform to generate a dataframe
        input:
            - transform and list
        returns:
            - df
        """
        return pd.DataFrame(transform,columns=features)


    

    def split_data(self,response_variable,split_ratio,get):
        """
        purpose:
            - splits the dataset into manageable portions
        input:
            - string,int and df
        returns:
            - df
        """
        X = self.df.drop(response_variable, axis=1)
        y = self.df[response_variable]
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
        
        


    

    def handle_missing_values(self,x):
        """
        purpose:
            - remove columns with x percentage of missing values
            - fill the missing values with the mean
        input:
            -df
        returns:
            - df
            - percentage of missing values
        """
        missing_percentage = round((self.df.isnull().sum().sum()/\
                reduce(lambda x, y: x*y, self.df.shape))*100,2)
        null_cols = self.df.isnull().sum().to_dict()
        for key,val in null_cols.items():
            if val/self.df.shape[0] > x:
                self.df.drop([key], axis=1)
            elif  val/self.df.shape[0] < x and val > 0 and self.df[key].dtype.kind in 'biufc':
                self.df.fillna(self.df[key].mean().round(1), inplace=True)
        return missing_percentage, self.df


   
    def top_x_column(self, x, column,color,online=False):
        """
        purpose:
            - to get the top x elements in a variable
        input:
            - string,int and df
        returns:
            - df
        """
        handsets_df = pd.DataFrame(columns = [column])
        handsets_df['type'] = self.df[column].to_list()
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
        


    
    def top_x_by_y_cols(self,col_1,col_2,x,y):
        """
        purpose:
            - gets the top y values of a x variable
        input:
            - string,int and df
        returns:
            - df
        """
        result_df = []
        by_manufacture = self.df.groupby(col_1,sort=True)
        values = self.top_x_column(self.df,x,col_1,"purple")


        for manufacturer, frame in by_manufacture:
            if manufacturer in values:
                result_df.append(frame.sort_values(by=[col_2], ascending=True)[col_2].head(y))
        return result_df
        


    
    def aggregation_cols(self,col_1,col_2,trim=False):
        """
        purpose:
            - returns the aggregations of two particular columns
            for comparison purpose
        input:
            - string and df
        returns:
            - df
        """
        grouped = self.df.groupby(col_1).agg({col_2: [min, max, mean]}) 
        grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
        if trim:
            return grouped.describe()
        return grouped


    
    def non_graphical_analysis(self,features,type_,opt,x_=1,y_=1):
        """
        purpose:
            - generates a non graphical summary of 
            a set of variables
        input:
            - string,int and df
        returns:
            - df
        """
        result = None
        if type_ == "univariate":
            for i,key in enumerate(features):
                if i == x:
                    result = pd.DataFrame(self.df[key].describe())
        elif type_ == "bivariate":
            for i,key in enumerate(features):
                if i == x_:
                    if opt=="regression":
                        y = self.df[features[y_]]
                        x = self.df[[key]]
                        x = sm.add_constant(x)
                        model = sm.OLS(y, x).fit()
                        result =  model.summary()
                    elif opt=="corr":
                        result = pd.DataFrame(self.df[[key,features[y]]].corr())
        elif type_ == "multivariate":
            result = pd.DataFrame(self.df[features].corr())
        return result
            
    def graphical_analysis(self,features,type_,opt,x=1,y=1):
        """
        purpose:
            - generates graphical of univariate or multivariate
        input:
            - string,int,list,df
        returns:
            - df
        """
        if type_ == "univariate":
            for i,key in enumerate(features):
                if i == x:
                    if opt == 'box':
                        return self.df.boxplot(column=[key], grid=False, color='black')
                    elif opt == 'hist':
                        return self.df.hist(column=[key], grid=False, edgecolor='black')
                    elif opt == 'curve':
                        return sns.kdeplot(self.df[key])
        if type_ == "bivariate":
            for i,key in enumerate(features):
                if i == x:
                    if opt == "scatter":
                        plt.scatter(self.df[features[x]], self.df[features[y]])
                        plt.title(f'{features[x]} vs {features[y]}')
                        plt.xlabel(f'{features[x]}')
                        plt.ylabel(f'{features[y]}')
                    
        


    
    def setup_pca(self,data,n):
        """
        purpose:
            - setting up of the data for pca analysis
        input:
            - int,df
        returns:
            - pca,df
        """
        pca = PCA(n)
        x_ = pca.fit_transform(data)
        return x_, pca




    def pca_analysis(self,features,type_,no,x_,component):
        """
        purpose:
            - performs a pca analysis on the data provided
        input:
            - string,int,list,df
        returns:
            - df
        """
        for i,key in enumerate(features):
            if i==x_:
                train = self.generate_transformation(self.generate_pipeline(type_),
                                            pd.DataFrame(self.split_data(self.df,key,0.3,"X_train")),
                                            type_,"number")
                test = self.generate_transformation(self.generate_pipeline(type_),
                                                pd.DataFrame(self.split_data(self.df,key,0.3,"X_test")),
                                                type_,"number")
                pca_train_results, pca_train = self.setup_pca(train, no)
                pca_test_results, pca_test = self.setup_pca(test, no)
                names_pcas = [f"PCA Component {i}" for i in range(1, 11, 1)]
                scree = pd.DataFrame(list(zip(names_pcas, pca_train.explained_variance_ratio_)), columns=["Component", "Explained Variance Ratio"])
                df = pd.DataFrame({'PCA':pca_train.components_[component], 'Variable Names':features})            
                df = df.sort_values('PCA', ascending=False)
                df2 = pd.DataFrame(df)
                df2['PCA']=df2['PCA'].apply(np.absolute)
                df2 = df2.sort_values('PCA', ascending=False)
                return df2
        return

    def categorize_based_on_deciles(self,df,features,x_):
        """
        purpose:
            - categorizes the data based on deciles
        input:
            - int,list and df
        returns:
            - df
        """
        for i,key in enumerate(features):
            if i==x_:
                self.df['decile_rank'] = pd.qcut(self.df[key], 10,labels = False)
                return self.df.groupby(['decile_rank']).sum()
        return


    def map_index_to_feature(self,index,features):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        for i,x in enumerate(features):
            if i == index:
                return x

   
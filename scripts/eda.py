#!/usr/bin/env python
# coding: utf-8
# author: Martin Luther Bironga
# date: 5/11/2022
import warnings
from xml.etree.ElementInclude import include
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from statistics import mean
import numpy as np
import statsmodels.api as sm


import matplotlib
plt.style.use('ggplot')

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


    
    def generate_transformation(self,pipeline,type_,value,trim=None,key=None):
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
            if trim:
                transformation=pipeline.fit_transform(pd.DataFrame(self.split_data(key,0.3,trim)).select_dtypes(include=value))
        elif type_ == "categorical":
            transformation=pipeline.fit_transform(self.df.select_dtypes(exclude=value))
            if trim:
                transformation=pipeline.fit_transform(pd.DataFrame(self.split_data(key,0.3,trim)).select_dtypes(exclude=value))
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
        
        


    

    def handle_missing_values_numeric(self, features):
        """
        this algorithm does the following
        - remove columns with x percentage of missing values
        - fill the missing values with the mean
        returns:
            - df
            - percentage of missing values
        """
        missing_percentage = round((self.df.isnull().sum().sum()/\
                reduce(lambda x, y: x*y, self.df.shape))*100,2)
        for key in features:
            self.df.fillna(self.df[key].mean().round(1), inplace=True)
        return missing_percentage, self.df

    def handle_missing_values_categorical(self,features):
        """
        this algorithm does the following
        - remove columns with x percentage of missing values
        - fill the missing values with the mode
        returns:
            - df
            - percentage of missing values
        """
        missing_percentage = round((self.df.isnull().sum().sum()/\
                reduce(lambda x, y: x*y, self.df.shape))*100,2)
        for key in features:
            self.df[key] = self.df[key].fillna(self.df[key].mode()[0])
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
        values = self.top_x_column(x,col_1,"purple")


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
                if i == x_:
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
                        fig,ax = plt.subplots()
                        ax.boxplot(self.df[key])
                        return fig
                    elif opt == 'hist':
                        fig,ax = plt.subplots()
                        ax.hist(self.df[key])
                        return fig
                    elif opt == 'curve':
                        fig,ax = plt.subplots()
                        ax.bar(self.df[key],height=2)
                        return fig
        if type_ == "bivariate":
            for i,key in enumerate(features):
                if i == x:
                    if opt == "scatter":
                        fig,ax = plt.subplots()
                        ax.scatter(self.df[features[x]], self.df[features[y]])
                        ax.set_title(f'{features[x]} vs {features[y]}')
                        ax.set_xlabel(f'{features[x]}')
                        ax.set_ylabel(f'{features[y]}')
                        return fig
        


    
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
                                            type_,"number",trim="X_train",key=key)
                test = self.generate_transformation(self.generate_pipeline(type_),
                                            type_,"number",trim="X_test",key=key)
                pca_train_results, pca_train = self.setup_pca(train, no)
                pca_test_results, pca_test = self.setup_pca(test, no)
                names_pcas = [f"PCA Component {i}" for i in range(1, 11, 1)]
                scree = pd.DataFrame(list(zip(names_pcas, pca_train.explained_variance_ratio_)), columns=["Component", "Explained Variance Ratio"])
                d = {'PCA':pca_train.components_[component], 'Variable Names':numerical_features[:x_]}
                df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
                df = df.sort_values('PCA', ascending=False)
                df2 = pd.DataFrame(df)
                df2['PCA']=df2['PCA'].apply(np.absolute)
                df2 = df2.sort_values('PCA', ascending=False)
                return df2
        return

    def categorize_based_on_deciles(self,features,x_):
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
                self.df['decile_rank'] = pd.qcut(self.df[key], 10,labels = False,duplicates='drop')
                return self.df.groupby(['decile_rank']).sum()
        return

    def fixing_outliers(self, column):
        """
        purpose: 
            - this function removes outliers from the data
        input:
            - df, int
        output:
            - df
        """
        self.df[column] = np.where(self.df[column] > self.df[column].quantile(0.95), 
                                  self.df[column].median(),self.df[column])
        
        return self.df[column]

    def map_index_to_feature(self,variable,features):
        """
        purpose:
            - generates transformations for the data
        input:
            - string,int and df
        returns:
            - transformation
        """
        for i,x in enumerate(features):
            if x == variable:
                return i


if __name__=="__main__":
    df = pd.read_csv("../data/telcom.csv")
    analyzer = EDA(df)
    numeric_pipeline = analyzer.generate_pipeline("numeric")
    numerical_features = analyzer.store_features("numeric","number")
    numeric_transformation = analyzer.generate_transformation(numeric_pipeline,"numeric","number")
    numeric_df = analyzer.frame_transforms(numeric_transformation,numerical_features)
    values = analyzer.top_x_column(10,"Handset Manufacturer","purple")
    fig,values = analyzer.top_x_column(10,"Handset Manufacturer","purple",online=True)
    values_ = analyzer.top_x_by_y_cols('Handset Manufacturer','Handset Type',3,5)
    aggregations = analyzer.aggregation_cols('MSISDN/Number','Total UL (Bytes)',True)
    analysis_1 = analyzer.non_graphical_analysis(numerical_features,"univariate",3)
    analysis_2 = analyzer.graphical_analysis(numerical_features,"univariate","curve",x=1)
    analysis_3 = analyzer.non_graphical_analysis(numerical_features,"multivariate",1,4)
    analysis_4 = analyzer.pca_analysis(numerical_features,"numeric",10,49,1)
    analysis_5 = analyzer.categorize_based_on_deciles(numerical_features,49)
    print(fig)


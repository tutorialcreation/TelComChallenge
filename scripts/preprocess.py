from functools import reduce

import pandas as pd


class Preprocess:
    def __init__(self, path):
        self.df = pd.read_excel(path, engine="openpyxl")

    def handle_missing_values(self, df, x):
        """
        this algorithm does the following
        - remove columns with x percentage of missing values
        - fill the missing values with the mean
        returns:
            - df
            - percentage of missing values
        """
        missing_percentage = round(
            (self.df.isnull().sum().sum() / reduce(lambda x, y: x * y, self.df.shape))
            * 100,
            2,
        )
        cols_fill, cols_out = [], []
        null_cols = self.df.isnull().sum().to_dict()
        for key, val in null_cols.items():
            if val / self.df.shape[0] > x:
                cols_out.append(key)
            elif val > 0 and self.df[key].dtype.kind in "biufc":
                cols_fill.append(key)
        self.df.drop(cols_out, axis=1)
        for i in self.df.columns:
            if i in cols_fill:
                self.df.fillna(self.df[i].mean().round(1), inplace=True)
        return missing_percentage, self.df

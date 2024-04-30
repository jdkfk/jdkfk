import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os   
import sys
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

class Dataset:
    
    def __init__(self,
                 dataframe:pd.DataFrame):
        self.df = dataframe.copy()
        print(self.df.describe())

    def describe_data(self,
                      to_txt:bool=False,
                      hist:bool=False,
                      path:str=r'./'):
        """
        data aggregation function.

        describe_data method provides the ability to calculate a number of aggregations
        for the columns that are contained in the Dataset dataframe. 

        Keyword Arguments:
            to_txt -- _description_ (default: {False})
            path -- _description_ (default: {r'./'})
        """
        def _aggregates():
            for col in self.df.columns:
                print(col)
                print(f'Count: {self.df[col].count()}')
                print(f'Data type: {self.df[col].dtype}')
                print(f'Non unique values: {self.df[col].nunique()}')
                print(f'Mode: {self.df[col].mode()}')
                if pd.api.types.is_numeric_dtype(self.df[col]): 
                    print(f'mean: {self.df[col].mean()}')
                    print(f'median: {self.df[col].median()}')
                    print(f'min: {self.df[col].min()}')
                    print(f'1st quartile: {self.df[col].quantile(.25)}')
                    print(f'2nd quartile: {self.df[col].quantile(.5)}')
                    print(f'3rd quartile: {self.df[col].quantile(.75)}')
                    print(f'max: {self.df[col].max()}')
                    print(f'skew: {self.df[col].skew()}')
                    print(f'kurtosis: {self.df[col].kurt()}')
                else:
                    if self.df[col].nunique()<10:
                        print(f'values: {self.df[col].value_counts()}')
                print('\n')
            
        def _histograms():
            
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    print(f'{col} histogram is being made')
                    fig, ax = plt.subplots()
                    ax = sns.histplot(x=self.df[col])
                    try:
                        fig.savefig(f'./figures/{col}_hist.pdf')
                    except:
                        os.mkdir(r'./figures')
                    del(fig)
                    del(ax)
        
        if to_txt:
            # Reroute standard output to a text file
            stdout_orig=sys.stdout
            with open('data_description.txt', 'w') as file:
                sys.stdout = file
                _aggregates()
            sys.stdout = stdout_orig
        else:
            _aggregates()
        
        if hist:
            _histograms()

    def interquartile_filter(
            self,
            col:str,
            path:str=r'./',
            hist:bool=False,
            factor:float=1.5,
            group_by:list=[]):
        
        """
        Interquartile_filter method filters using the interquartile method the passed column. 

        Interquartile_filter method calculates the interquartile outliers from the column passed 
        as the argument col in order to identify outliers of the regarding parameter.

        Arguments:
            col -- name of the column to be filtered
            path -- path on which the histograms are stored (optional)
            hist -- set to True in order to generate histogram
            factor -- factor applied to the standard deviation in order to clasify the outliers
            group_by -- column for the grouped calculation
        """

        def _histogram(col):
            if hist:
                fig, ax = plt.subplots()

                ax = sns.histplot(data=self.df, x=col, hue=f'{col}_iqr_outliers')
            
                try:
                    fig.savefig(f'./figures/{col}_hist.pdf')
                except:
                    os.mkdir(r'./figures')
                    fig.savefig(os.path.join(path,f'{col}_iqr_outlier_hist.pdf'))
                del(fig)

        def _outliers(
            col,
            group_by:str=''
        ):
            outlier_col = f'{col}_iqr_outliers'
            if outlier_col not in self.df.columns:
                self.df[outlier_col] = False
            
            if group_by=='':
                perc25 = self.df[col].describe()['25%']
                perc75 = self.df[col].describe()['75%']
                iqr = (perc75 - perc25) * factor
                
                self.df.loc[
                    ((self.df[f'{col}']<perc25-iqr) |
                    (self.df[f'{col}']>perc75+iqr)) &
                    (self.df[f'{col}']!=0),
                    outlier_col] = True
            elif group_by not in self.df.columns:
                print('The grouping column does not exist within the dataframe')
            else:
                for val in self.df[group_by].unique():
                    perc25 = self.df.loc[self.df[group_by]==val,:][col].describe()['25%']
                    perc75 = self.df.loc[self.df[group_by]==val,:][col].describe()['75%']
                    iqr = (perc75 - perc25) * factor
                    
                    self.df.loc[
                        (((self.df[f'{col}']<perc25-iqr) |
                        (self.df[f'{col}']>perc75+iqr)) &
                        (
                            (self.df[f'{col}']!=0) |
                            (self.df[col].isna())
                        ) & 
                        (self.df[group_by]==val)),
                        outlier_col] = True


        
        # Check if column is defined and loop over every numeric column if not.
        if col=='':
            for column in [col for col in self.df.columns if col in ['int64', 'float64']]:
                _outliers(column,group_by)
                _histogram(column)
        
        else:
            _outliers(col, group_by=group_by)
            _histogram(col)

    def normalize(
        self,
        col:str,
        weight:float=1.0):
        """
        Normalize columns from the containing dataframe.

        The normalize method from Dataset class normalizes the columns
        passed as arguments and creates a new normalized column appending 
        the string _norm at the end of the name of the column. 

        Arguments:
            col -- name of the column to be normalized
        """       
        if col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[f'{col}_norm'] = self.df[col]/max(
                    abs(self.df[col].max()),
                    abs(self.df[col].min())) * weight

    def k_means_cluster(
        self,
        cols:list,
        col_weights:list=[],
        clusters:int=3):
        """
        k_means_cluster performs clustering of columns in a number of clusters

        k_means_cluster methods provides the ability to perform a k-means clustering on the 
        columns passed as an argument cols in a number of clusters equal to argument clusters.

        Arguments:
            cols -- list of clumns to be clustered

        Keyword Arguments:
            n_clusters -- number of clusters (default: {3})

        Returns:
            New column with the clusters result
        """
        norm_cols = [f'{col}_norm' for col in cols]
        cluster_column_name = '_'.join(cols)
        
        if (len(cols) != len(col_weights) & len(col_weights) > 0) | (len(col_weights) == 0):
            if len(col_weights)>0:
                print('Column weights has been ignored due to a length mismatch.')
            col_weights = [1 for col in col_weights]

        weight_dict = dict(zip(cols,col_weights))

        for col in cols:
            self.normalize(col,weight=weight_dict[col])

        _X = self.df.loc[:,norm_cols]
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(_X)
        self.df[f'{cluster_column_name}_cluster']=kmeans.predict(_X)

    def linear_regression(
        self,
        X_cols:list,
        y_col:str):
        """
        Linear regression for this library.

        Perform a linear regression as a function of X_cols columns and creates 
        a new column with the estimation based on the resulting model. 

        Arguments:
            X_cols -- linear regression variables
            y_col -- target column
        """
        #X_train,y_train,X_test,y_test = train_test_split()
        model = LinearRegression(fit_intercept=True)
        all_cols = X_cols + [y_col]
        _df=self.df.loc[:,all_cols].dropna()
        X = _df.loc[:,X_cols]
        y = _df[y_col]
        model.fit(X,y)
        _df[f'{y_col}_estim_linear'] = model.predict(X)
        self.df = pd.merge(
            self.df,
            _df.loc[:,f'{y_col}_estim_linear'],
            right_index=True,
            left_index=True,
            how='left')
        
    def error(
        self,
        col:str):
        # from sklearn.metrics import mean_absolute_error y programar la función. Estudiar si pudiesen ser necesarias más funciones.
        #mean_absolute_error(all.df.loc[~all.df.fb_climb.isna(),:].fb_climb,all.df.loc[~all.df.fb_climb.isna(),:].fb_climb_estim_linear)
        res = mean_absolute_error(
            self.df[col],
            self.df[f'{col}_estim_linear']
        )
        return mean_absolute_error(self.df[col])
    
    def dt_classifier(
        self,
        X_cols:list,
        y_col:str):
        """
        Decision tree classifier for this library.

        Perform a decision tree classification as a function of X_cols columns and creates 
        a new column with the estimation based on the resulting model. 

        Arguments:
            X_cols -- classification variables
            y_col -- target column
        """
        #X_train,y_train,X_test,y_test = train_test_split()
        model = DecisionTreeClassifier(random_state=0)
        all_cols = X_cols + [y_col]
        _df=self.df.loc[:,all_cols].dropna()
        X = _df.loc[:,X_cols]
        y = _df[y_col]
        model.fit(X,y)
        _df[f'{y_col}_dt_clasif'] = model.predict(X)
        self.df = pd.merge(
            self.df,
            _df.loc[:,f'{y_col}_dt_clasif'],
            right_index=True,
            left_index=True,
            how='left')


class Dataset:
    
    import pandas as pd
    def __init__(self, dataframe:pd.DataFrame):
        self.df = dataframe.copy()
        print(self.df.describe())
    


    def describe_data(self, 
                      to_txt:bool=False, 
                      hist:bool=False, 
                      path:str=r'./'):
        """
        data aggregation

        Keyword Arguments:
            to_txt -- _description_ (default: {False})
            path -- _description_ (default: {r'./'})
        """
        def _aggregates():
            import pandas as pd
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
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            import os
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
            import sys
            stdout_orig=sys.stdout
            with open('data_description.txt', 'w') as file:
                sys.stdout = file
                _aggregates()
            sys.stdout = stdout_orig
        else:
            _aggregates()
        
        if hist:
            _histograms()




    #interquartile test for sat @ TOFF
    def interquartile_filter(
            self, 
            col:str, 
            hist:bool=False, 
            factor:float=1.5):
        
        """
        _summary_
        """
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        perc25 = self.df[col].describe()['25%']
        perc75 = self.df[col].describe()['75%']
        iqr = (perc75 - perc25) * factor

        self.df[f'{col}_iqr_outliers'] = False
        self.df.loc[
            ((self.df[f'{col}']<perc25-iqr) |
            (self.df[f'{col}']>perc75+iqr)) &
            (self.df[f'{col}']!=0),
            f'{col}_iqr_outliers'] = True

        if hist:
            fig, ax = plt.subplots()

            ax = sns.histplot(data=self.df, x=col, hue=f'{col}_iqr_outliers')
            fig.savefig(f'{col}_iqr_outlier_hist.pdf')

            del(fig)



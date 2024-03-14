import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def hist(data:pd.DataFrame, col:str, hue_col:str, type:str='hist'):
    fig, ax = plt.subplots(figsize=(6,6))
    
    if type == 'hist':
        ax = sns.histplot(data[col])
    elif type == 'kde':
        ax = sns.kdeplot(data[col])
    return fig, ax

# def hist(_df,param,group_col='AC_TYPE',outlier=False):
#     fig = plt.figure(figsize=(10,20))
#     n_graphs = _df[group_col].nunique()
#     gs = fig.add_gridspec(n_graphs,1)
#     i=0
#     for group in np.sort(_df[group_col].unique()):
#         ax = plt.subplot(gs[i,0])
#         i+=1
#         ax.hist(_df.loc[_df[group_col]==group][param],edgecolor='black',bins=30)
#         if outlier:
#             ax.scatter(x=_df.loc[(_df[f'{param}_filter']) & (_df[group_col]==group),:][param],y=[1]*len(_df.loc[(_df[f'{param}_filter']) & (_df[group_col]==group),:]),color='black',alpha=0.5)
#         ax.set_title(group)
#         ax.set_xlim(left=0,right=_df[param].max())
    
#     fig.suptitle(param)
#     plt.tight_layout()
#     plt.show()
#     plt.close()
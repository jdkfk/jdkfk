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

import pandas as pd
import numpy as np
import os
import sys

pd.options.display.max_rows = 15
pd.options.display.max_columns = 15
pd.set_option("display.width",150)

print('hello')

os.getcwd()
list_factors=['sector', 'momentum','quality','growth','vol','value','size']

df = pd.read_csv('stock_data_actual_dates.csv').iloc[:,1:]

df.groupby('date').count().plot()

df

#############################
# first build a light weight backtester
#############################

n = 5

df_2 = add_quintiles_as_new_col(df,
                             col_name = 'momentum',
                             new_col_name = 'mom_q',
                             groupby_col_name = 'date',
                                n = 5)

df_2.groupby('date').mom_q.value_counts()

avg_rets = df_2.groupby(['mom_q','date'])['fwd_returns'].mean().unstack().T

def calc_sharpe(x,
                n=12):
    return x.mean()* np.sqrt(n) / x.std()

avg_rets.apply(calc_sharpe)

np.cumprod(1+(avg_rets.unstack().T*0.01)).plot()





#############################

def add_quintiles_as_new_col(df,
                             col_name,
                             new_col_name,
                             groupby_col_name = 'date',
                             n=10):
    """
    
    :param col_name: 
    :param n: 
    :param new_col_name: 
    :param groupby_col_name: 
    :return: 
    """
    _df = df.copy()

    _df[new_col_name] = _df.groupby(groupby_col_name)[col_name].transform(lambda x: pd.qcut(
        x, q = n, labels = np.arange(1,n+1))).astype(float)
    return _df



#############################
#
#############################





#############################
#
#############################






#############################
#
#############################





#############################
#
#############################




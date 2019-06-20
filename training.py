
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
sns.set()

pd.options.display.max_rows = 15
pd.options.display.max_columns = 15
pd.set_option("display.width",150)

print('hello')

os.getcwd()
list_factors=['sector', 'momentum','quality','growth','vol','value','size']

df = pd.read_csv('stock_data_actual_dates.csv').iloc[:,1:]

df.groupby('date').count().plot()

#############################
# first build a light weight backtester
#############################

############################


sig = SignalUnivariateStudy(data_df = df,
                            factor_name = 'momentum',
                            neutralizer_column = 'sector',
                            order = 'asc',
                            n = 10)

sig.data_df.groupby(['date','sector']).vol_SN.describe()

sig.wealth.plot()

df['fwd_returns'].mean()

print(sig.stats)

pd.DataFrame(sig.stats)

############################



############################

def calc_stats(_ret_series,
               n=12 # for monthly
                 ):
    """
    
    Parameters
    ----------
    _ret_series

    Returns
    -------

    """
    _stats = {}

    adj_factor = np.sqrt(n)
    num_obs = _ret_series.shape[0]

    _stats['rets'] = _ret_series.mean() * n
    _stats['vol'] = _ret_series.std() * adj_factor
    _stats['sharpe']=_stats['rets'] /_stats['vol']
    _stats['tstat'] = _stats['sharpe'] * np.sqrt(num_obs) / adj_factor
    _stats['start_dt'] = _ret_series.index[0]
    _stats['end_dt'] = _ret_series.index[-1]
    #import pdb; pdb.set_trace()
    return _stats




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
        x, q = n, labels = np.arange(1,n+1), precision=0)).astype(str)
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




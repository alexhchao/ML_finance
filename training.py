
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

df

#############################
# first build a light weight backtester
#############################
order = 'ascending'
n = 5

df_2 = add_quintiles_as_new_col(df,
                             col_name = 'momentum',
                             new_col_name = 'mom_q',
                             groupby_col_name = 'date',
                                n = 5)

df_2['mom_q'] = df_2.groupby('date')['momentum'].transform(lambda x: pd.qcut(
        x, q=n, labels= [str(x).split('.')[0] for x in np.arange(1,n+1)]  )).astype(
            str).apply(lambda x: x.split('.')[0])

x = df_2.groupby(['date','sector']).momentum.percentile_rank()

df = add_sector_neutral_column(df = df,
                          col_to_neutralize = 'momentum',
                          neutralized_col_name=None,
                          agg_col_names = ['date','sector'])


df.query("date == '2006-01-31'").query("momentum.notnull()", engine = 'python').query("sector==2.0")


x = df_2.groupby(['date','sector']).momentum.rank(pct=True)

x.dropna()


df_2['mom_q'].replace('nan',np.NaN, inplace = True)

#df_2['mom_q'] = df_2['mom_q'].apply(lambda x: x.split('.')[0])

df_2.dtypes

df_2.groupby('date').mom_q.value_counts()


# Long minus short basket
if order = 'ascending':
    avg_rets['LS - {} minus {}'.format()]


avg_rets.apply(calc_sharpe)

_ret = avg_rets.iloc[:,0]

calc_stats(_ret/100)


rets = df_2.groupby(['mom_q','date'])['fwd_returns'].mean().unstack().T.shift(1) * 0.01
#rets.loc[:,'1']

# okay this gets stats for each basket
all_stats = rets.apply(lambda x: pd.Series(calc_stats(x)))

wealth = np.cumprod(1+rets)

wealth.plot()


#{calc_stats(avg_rets.loc[:,x]) for x in list(avg_rets.columns)}



_ret.index[0]
_ret.index[-1]


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




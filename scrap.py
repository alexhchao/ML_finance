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





"""

Packages to use :

tsfresh

tsfel https://tsfel.readthedocs.io/en/latest/

sktime

feature tools : https://docs.featuretools.com/en/stable/automated_feature_engineering/handling_time.html

Cesium http://cesium-ml.org/docs/feature_table.html

Feature Tools for advacned fewatures `https://github.com/Featuretools/predict-remaining-useful-life/blob/master/Advanced%20Featuretools%20RUL.ipynb


"""

import pandas as pd
import tsfresh




def features_time_basic(dfraw, fname):
	df = copy.deepcopy(dfraw)
	df['date'] = pd.to_datetime(df['date'])
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month
	df['week'] = df['date'].dt.week
	df['day'] = df['date'].dt.day
	df['dayofweek'] = df['date'].dt.dayofweek	
	return df


    
    
def features_lag(df, fname):
    out_df = df[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    ###############################################################################
    # day lag 29~57 day and last year's day lag 1~28 day 
    day_lag = df.iloc[:,-28:]
    day_year_lag = df.iloc[:,-393:-365]
    day_lag.columns = [str("lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    day_year_lag.columns = [str("lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    # Rolling mean(3) and (7) and (28) and (84) 29~57 day and last year's day lag 1~28 day 
    rolling_3 = df.iloc[:,-730:].T.rolling(3).mean().T.iloc[:,-28:]
    rolling_3.columns = [str("rolling3_lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    rolling_3_year = df.iloc[:,-730:].T.rolling(3).mean().T.iloc[:,-393:-365]
    rolling_3_year.columns = [str("rolling3_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_7 = df.iloc[:,-730:].T.rolling(7).mean().T.iloc[:,-28:]
    rolling_7.columns = [str("rolling7_lag_{}_day".format(i)) for i in range(29,57)] # Rename columns
    rolling_7_year = df.iloc[:,-730:].T.rolling(7).mean().T.iloc[:,-393:-365]
    rolling_7_year.columns = [str("rolling7_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_28 = df.iloc[:,-730:].T.rolling(28).mean().T.iloc[:,-28:]
    rolling_28.columns = [str("rolling28_lag_{}_day".format(i)) for i in range(29,57)]
    rolling_28_year = df.iloc[:,-730:].T.rolling(28).mean().T.iloc[:,-393:-365]
    rolling_28_year.columns = [str("rolling28_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    rolling_84 = df.iloc[:,-730:].T.rolling(84).mean().T.iloc[:,-28:]
    rolling_84.columns = [str("rolling84_lag_{}_day".format(i)) for i in range(29,57)]
    rolling_84_year = df.iloc[:,-730:].T.rolling(84).mean().T.iloc[:,-393:-365]
    rolling_84_year.columns = [str("rolling84_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]

    # monthly lag 1~18 month
    month_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly = df.iloc[:,-28*i:].T.sum().T
            month_lag["monthly_lag_{}_month".format(i)] = monthly
        else:
            monthly = df.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_lag["monthly_lag_{}_month".format(i)] = monthly

    # combine day lag and monthly lag
    out_df = pd.concat([out_df, day_lag], axis=1)
    out_df = pd.concat([out_df, day_year_lag], axis=1)
    out_df = pd.concat([out_df, rolling_3], axis=1)
    out_df = pd.concat([out_df, rolling_3_year], axis=1)
    out_df = pd.concat([out_df, rolling_7], axis=1)
    out_df = pd.concat([out_df, rolling_7_year], axis=1)
    out_df = pd.concat([out_df, rolling_28], axis=1)
    out_df = pd.concat([out_df, rolling_28_year], axis=1)
    out_df = pd.concat([out_df, rolling_84], axis=1)
    out_df = pd.concat([out_df, rolling_84_year], axis=1)
    out_df = pd.concat([out_df, month_lag], axis=1)

    ###############################################################################
    # dept_id
    group_dept = df.groupby("dept_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    dept_day_lag = group_dept.iloc[:,-28:]
    dept_day_year_lag = group_dept.iloc[:,-393:-365]
    dept_day_lag.columns = [str("dept_lag_{}_day".format(i)) for i in range(29,57)]
    dept_day_year_lag.columns = [str("dept_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_dept_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_dept = group_dept.iloc[:,-28*i:].T.sum().T
            month_dept_lag["dept_monthly_lag_{}_month".format(i)] = monthly_dept
        elif i >= 7 and i < 13:
            continue
        else:
            monthly = group_dept.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_dept_lag["dept_monthly_lag_{}_month".format(i)] = monthly_dept
    # combine out df
    out_df = pd.merge(out_df, dept_day_lag, left_on="dept_id", right_index=True, how="left")
    out_df = pd.merge(out_df, dept_day_year_lag, left_on="dept_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_dept_lag, left_on="dept_id", right_index=True, how="left")

    ###############################################################################       
    # cat_id
    group_cat = df.groupby("cat_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    cat_day_lag = group_cat.iloc[:,-28:]
    cat_day_year_lag = group_cat.iloc[:,-393:-365]
    cat_day_lag.columns = [str("cat_lag_{}_day".format(i)) for i in range(29,57)]
    cat_day_year_lag.columns = [str("cat_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_cat_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_cat = group_cat.iloc[:,-28*i:].T.sum().T
            month_cat_lag["cat_monthly_lag_{}_month".format(i)] = monthly_cat
        elif i >= 7 and i < 13:
            continue
        else:
            monthly_cat = group_cat.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_cat_lag["dept_monthly_lag_{}_month".format(i)] = monthly_cat

    # combine out df
    out_df = pd.merge(out_df, cat_day_lag, left_on="cat_id", right_index=True, how="left")
    out_df = pd.merge(out_df, cat_day_year_lag, left_on="cat_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_cat_lag, left_on="cat_id", right_index=True, how="left")

    ###############################################################################
    # store_id
    group_store = df.groupby("store_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    store_day_lag = group_store.iloc[:,-28:]
    store_day_year_lag = group_store.iloc[:,-393:-365]
    store_day_lag.columns = [str("store_lag_{}_day".format(i)) for i in range(29,57)]
    store_day_year_lag.columns = [str("store_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_store_lag = pd.DataFrame({})
    for i in range(1,19):
        if i == 1:
            monthly_store = group_store.iloc[:,-28*i:].T.sum().T
            month_store_lag["store_monthly_lag_{}_month".format(i)] = monthly_store
        elif i >= 7 and i <13:
            continue
        else:
            monthly_store = group_store.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_store_lag["store_monthly_lag_{}_month".format(i)] = monthly_store

    # combine out df
    out_df = pd.merge(out_df, store_day_lag, left_on="store_id", right_index=True, how="left")
    out_df = pd.merge(out_df, store_day_year_lag, left_on="store_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_store_lag, left_on="store_id", right_index=True, how="left")

    ###############################################################################
    # state_id
    group_state = df.groupby("state_id").sum()
    # day lag 29~57 day and last year's day lag 1~28 day 
    state_day_lag = group_state.iloc[:,-28:]
    state_day_year_lag = group_state.iloc[:,-393:-365]
    state_day_lag.columns = [str("state_lag_{}_day".format(i)) for i in range(29,57)]
    state_day_year_lag.columns = [str("state_lag_{}_day_of_last_year".format(i)) for i in range(1,29)]
    # monthly lag 1~18 month
    month_state_lag = pd.DataFrame({})
    for i in range(1,13):
        if i == 1:
            monthly_state = group_state.iloc[:,-28*i:].T.sum().T
            month_state_lag["state_monthly_lag_{}_month".format(i)] = monthly_state
        elif i >= 7 and i < 13:
            continue
        else:
            monthly_state = group_state.iloc[:, -28*i:-28*(i-1)].T.sum().T
            month_state_lag["state_monthly_lag_{}_month".format(i)] = monthly_state

    # combine out df
    out_df = pd.merge(out_df, state_day_lag, left_on="state_id", right_index=True, how="left")
    out_df = pd.merge(out_df, state_day_year_lag, left_on="state_id", right_index=True, how="left")
    out_df = pd.merge(out_df, month_state_lag, left_on="state_id", right_index=True, how="left")

    ###############################################################################
    # category flag
    col_list = ['dept_id', 'cat_id', 'store_id', 'state_id']

    df_cate_oh = pd.DataFrame({})
    for i in col_list:
        df_oh = pd.get_dummies(df[i])
        df_cate_oh = pd.concat([df_cate_oh, df_oh], axis=1)

    out_df = pd.concat([out_df, df_cate_oh], axis=1)

    out_df.to_parquet(fname) 
    # return out_df
  
"""
def basic_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.week
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    return df[['year', 'month', 'week', 'day', 'dayofweek']]
"""

def features_mean(df):
    pass


def features_identity(df):
    return df

 def features_rolling(df):
    created_cols = []

    len_shift = 28
    for i in [7,14,30,60,180]:
        print('Rolling period:', i)
        df['rolling_mean_'+str(i)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(len_shift).rolling(i).mean())
        df['rolling_std_'+str(i)]  = df.groupby(['id'])['demand'].transform(lambda x: x.shift(len_shift).rolling(i).std())
        created_cols.append('rolling_mean_'+str(i))
        created_cols.append('rolling_std_'+str(i))

    # Rollings
    # with sliding shift
    for len_shift in [1,7,14]: 
        print('Shifting period:', len_shift)
        for len_window in [7,14,30,60]:
            col_name = 'rolling_mean_tmp_'+str(len_shift)+'_'+str(len_window)
            df[col_name] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(len_shift).rolling(len_window).mean())
            created_cols.append(col_name)

    return df[created_cols]



def features_lag(df):
    created_cols = []
    lag_days = [col for col in range(28, 28+15)]
    for lag_day in lag_days:
        created_cols.append('lag_' + str(lag_day))
        df['lag_' + str(lag_day)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(lag_day))

    return df[created_cols]
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
'''sales_eval=pd.read_csv('sales_train_validation.csv')
cal=pd.read_csv('calendar.csv')
sales=pd.read_csv('sell_prices.csv')'''
def features_time_basic(cal):
	cal['date'] = pd.to_datetime(cal['date'])
	cal['year'] = cal['date'].dt.year
	cal['month'] = cal['date'].dt.month
	cal['week'] = cal['date'].dt.week
	cal['day'] = cal['date'].dt.day
	cal['dayofweek'] = cal['date'].dt.dayofweek
	date_cols=cal[['wday','month','year','week','day']]
	scaler = MinMaxScaler(feature_range=(-0.5,0.5))
	scaler.fit(date_cols)
	transformed=scaler.transform(date_cols)
	new_cal=pd.DataFrame(transformed,columns=date_cols.columns)
	embeded=cal[['event_name_1','event_type_1','event_name_2','event_type_2']]
	unique=embeded['event_name_1'].unique()
	unique_vals=np.append(unique,embeded['event_name_2'].unique())
	df_event_name=pd.DataFrame(unique_vals)
	array_nm=df_event_name[0].unique()
	list_array=list(array_nm)
	vocab_size = 50
	encoded_docs = [one_hot(str(d), vocab_size) for d in list_array]
	max_length = 2
	embedded_event_name = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	unique=embeded['event_type_1'].unique()
	unique_vals=np.append(unique,embeded['event_type_2'].unique())
	df_event_type=pd.DataFrame(unique_vals)
	array_nm=df_event_type[0].unique()
	list_array=list(array_nm)
	vocab_size = 50
	encoded_docs = [one_hot(str(d), vocab_size) for d in list_array]
	max_length = 1
	embedded_event_type = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	values=cal[['snap_CA','snap_TX','snap_WI']]
	return new_cal,embedded_event_name,embedded_event_type,values
def features_normalize(sales,cal):
    sales['dept_id']=str(sales['item_id'])
    def remove(x):
        value=x.split('_')
        dept=value[0]+"_"+value[1]
        return dept
    sales['dept']=sales['item_id'].apply(remove)
    price_sum=pd.DataFrame(sales.groupby('wm_yr_wk')['sell_price'].sum())
    scaler = MinMaxScaler()
    scaler.fit(price_sum)
    price_sum['normalized_price_time']=scaler.transform(price_sum)
    price_sum.drop('sell_price',axis=1,inplace=True)
    sellprice_sum=pd.DataFrame(sales.groupby('dept')['sell_price'].unique())
    sellprice_sum.reset_index(inplace=True)
    def recive(v):
    	sum=0
    	for i in v:
            sum+=i
    	return round(sum,2)
    sellprice_sum['summed_sellprice']=sellprice_sum['sell_price'].apply(recive)
    scaler = MinMaxScaler()
    sellprice_sum.set_index(keys='dept',inplace=True)
    sellprice_sum.drop('sell_price',inplace=True,axis=1)
    scaler.fit(sellprice_sum)
    sellprice_sum['normalized_price_dept']=scaler.transform(sellprice_sum)
    price_sum.reset_index(inplace=True)
    price_sum=pd.merge(price_sum,cal[['date','wm_yr_wk']],on='wm_yr_wk',how='left')
    price_sum.drop(['wm_yr_wk'],axis=1,inplace=True)
    sellprice_sum.reset_index(inplace=True)
    sellprice_sum.drop(['dept'],axis=1,inplace=True)
    return price_sum,sellprice_sum
def features_lag(train_df):
    TARGET='sales'
    index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
    train_df_1 = pd.melt(train_df,id_vars = index_columns,var_name = 'd',value_name = TARGET)
    sales_lag = train_df_1[['id','d',TARGET]]
    #lag=1
    i=1
    print('Shifting:', i)
    sales_lag['lag_'+str(i)] = sales_lag.groupby(['id'])[TARGET].transform(lambda x: x.shift(i))
    sales_lag=pd.concat([sales_lag,train_df_1['item_id']],axis=1)
    return sales_lag
def features_rolling(train_df):
    TARGET='sales'
    index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
    sales_eval_2 = pd.melt(train_df,id_vars = index_columns,var_name = 'd',value_name = TARGET)
    rolling_7_28 = sales_eval_2[['id','d','sales']]
    #start_time = time.time()
    for i in [7,28]:
        print('Rolling period:', i)
        rolling_7_28['rolling_mean_'+str(i)] = rolling_7_28.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).mean())
        rolling_7_28['rolling_std_'+str(i)]  = rolling_7_28.groupby(['id'])[TARGET].transform(lambda x: x.shift(1).rolling(i).std())
    rolling_7_28=pd.concat([rolling_7_28,sales_eval_2['item_id']],axis=1)
    return rolling_7_28
def features_embed(train_df):
    index_cols=['item_id','dept_id','cat_id','store_id','state_id']
    unique_vals=[]
    for i in index_cols:
        unique_vals.append(train_df[i].unique())
    train_df_categorical=pd.DataFrame(unique_vals).transpose()
    train_df_categorical.columns=['item_id','dept_id','cat_id','store_id','state_id']
    train_df_categorical=train_df_categorical.astype(str)
    for i in train_df_categorical.columns:
        v=list(train_df_categorical[i].dropna())
        vocab_size=len(v)
        encoded_docs = [one_hot(x, vocab_size) for x in v]
        max_length = 1
        padded_docs1 = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        train_df_categorical[i+"_encoded"]=padded_docs1
    return train_df_categorical
  '''
  
  
  
  
  
  
  

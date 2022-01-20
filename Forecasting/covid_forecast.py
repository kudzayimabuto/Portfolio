import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
from datetime import datetime, timedelta
# prophet model 
from fbprophet import Prophet
# prophet preformance
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import os
from pydlm import dlm, trend, seasonality, autoReg
import warnings
warnings.filterwarnings('ignore')

now = datetime.now()
file_name = 'covid_data.csv'
df = pd.read_csv(file_name)
df = df.loc[df['location'] == 'South Africa']
df = df.loc[df['date'] >= '2020-03-06']
df['date'] = pd.to_datetime(df['date'])
df_ds = df[['date', 'new_deaths']]
df_ds.columns = ["ds", "y"]
df_ds["y"] = df_ds["y"]


def plot_colname(dataf, column_name):
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("husl", 3)

    sns.lineplot(x='date',
                 y=column_name,
                 data=dataf,
                 ax=ax1,
                 ci=None
                 )
    fig.autofmt_xdate()
    plt.xlabel(column_name + '_per_day',
               fontweight='bold',
               color='k',
               fontsize='17',
               horizontalalignment='center')


plot_columns = ['total_cases',
                'new_cases',
                'total_deaths',
                'new_deaths',
                'total_cases_per_million',
                'new_cases_per_million',
                'total_deaths_per_million',
                'new_deaths_per_million',
                'total_tests', 'new_tests',
                'total_tests_per_thousand',
                'new_tests_per_thousand'
                ]

for i in plot_columns:
    plot_colname(df, i)


def profet(df, period):
    model = Prophet(changepoint_prior_scale=0.8, daily_seasonality=False,
                    yearly_seasonality=False, weekly_seasonality=True)
    model.fit(df)
    build_forecast = model.make_future_dataframe(periods=period, freq='D')
    forecast = model.predict(build_forecast)
    df_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    df_forecast[["yhat", "yhat_lower", "yhat_upper"]
                ] = df_forecast[["yhat", "yhat_lower", "yhat_upper"]]-1

    mask = df_ds.ds.values.tolist()
    df_forecast = df_forecast[~df_forecast['ds'].isin(mask)]
    #df_forecast[df_forecast < 0] = 0
    return df_forecast


def baysian(df, period):

    time_series = df.y.values
    # A linear trend
    linear_trend = trend(degree=1, discount=0.99, name='linear_trend', w=100)
    # A seasonality
    seasonal52 = seasonality(period=52, discount=0.99,
                             name='seasonal52', w=1.0)
    seasonal7 = seasonality(period=7, discount=0.99, name='seasonal7', w=1.0)
    # Build a simple dlm
    simple_dlm = dlm(time_series) + linear_trend + seasonal52 + seasonal7
    # Fit the model
    simple_dlm.fit()

    #predictions = np.exp(np.asarray(simple_dlm.predictN(date=60, N=period+1)[0]))-1
    predictions = np.asarray(simple_dlm.predictN(date=60, N=period+1)[0])
    upper = predictions+1.96*np.sqrt(np.var(predictions))
    lower = predictions-1.96*np.sqrt(np.var(predictions))

    start = df_ds.ds.max()
    end = (df_ds.ds.max() + timedelta(days=period+1))
    date_generated = [start + timedelta(days=x)
                      for x in range(0, (end-start).days)]

    new_date_df = pd.DataFrame(date_generated, columns=['date'])
    new_date_df['predictions'], new_date_df["upper_limit"], new_date_df["lower_limit"] = predictions, upper, lower
    new_date_df = new_date_df.drop(new_date_df.index[0])

    return new_date_df


prof_forecast = profet(df_ds, 90)
bays_forecast = baysian(df_ds, 90)
prof_forecast.columns = ['date', 'yhat', 'yhat_lower', 'yhat_upper']
df_final = prof_forecast.merge(bays_forecast, on='date', how='inner')

df_final['combined_upper'], df_final['combined_pred'], df_final['combined_lower'] = (
    df_final.yhat_upper+df_final.upper_limit)/2, (df_final.yhat+df_final.predictions)/2, (df_final.yhat_lower+df_final.lower_limit)/2
combined_forecast = df_final[[
    'date', 'combined_upper', 'combined_pred', 'combined_lower']]
combined_forecast.columns = [
    'date', 'upper_limit', 'prediction', 'lower_limit']
num = combined_forecast._get_numeric_data()
num[num < 0] = 0

fig = plt.figure(figsize=(16, 9))

ax1 = fig.add_subplot(111)

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
ax1.grid(False)
sns.set_palette("husl", 2)

data1 = combined_forecast.melt(
    'date', var_name='New covid cases',  value_name='new_cases')

sns.lineplot(x='date', y='new_cases', data=data1,
             ax=ax1, ci=None, hue='New covid cases')
fig.autofmt_xdate()
plt.xlabel('composite forecast', fontweight='bold', color='k',
           fontsize='17', horizontalalignment='center')
(df_ds.y.sum() + combined_forecast.prediction.sum()).round(0)
print(datetime.now() - now)
plt.savefig('forecast_' + now + '_.png')

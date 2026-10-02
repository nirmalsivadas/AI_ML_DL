from pathlib import Path

import numpy as np
import pandas as pd
import datetime

base_dir = Path(__file__).resolve().parent
date = pd.read_csv(base_dir / 'files' / 'orders.csv')
time = pd.read_csv(base_dir / 'files' / 'messages.csv')

print(date.head())
print(time.head())

print(date.info())
print(time.info())

# Converting to datetime datatype
date['date'] = pd.to_datetime(date['date'])

print(date.info())

date['date_year'] = date['date'].dt.year

print(date.sample(5))

date['date_month_no'] = date['date'].dt.month

print(date.head())

date['date_month_name'] = date['date'].dt.month_name()

print(date.head())

date['date_day'] = date['date'].dt.day

print(date.head())

# day of week
date['date_dow'] = date['date'].dt.dayofweek

print(date.head())

# day of week - name

date['date_dow_name'] = date['date'].dt.day_name()

print(date.drop(columns=['product_id','city_id','orders']).head())

# is weekend?

date['date_is_weekend'] = np.where(date['date_dow_name'].isin(['Sunday', 'Saturday']), 1,0)

print(date.drop(columns=['product_id','city_id','orders']).head())

date['date_week'] = date['date'].dt.isocalendar().week

print(date.drop(columns=['product_id','city_id','orders']).head())

date['quarter'] = date['date'].dt.quarter

print(date.drop(columns=['product_id','city_id','orders']).head())

date['semester'] = np.where(date['quarter'].isin([1,2]), 1, 2)

print(date.drop(columns=['product_id','city_id','orders']).head())

today = datetime.datetime.today()

print(today)

today - date['date']

(today - date['date']).dt.days

# Months passed (approximated as 30-day months for compatibility)

print(np.round((today - date['date']) / pd.Timedelta(days=30), 0))

print(time.info())

# Converting to datetime datatype
time['date'] = pd.to_datetime(time['date'])

print(time.info())

time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second

print(time.head())

time['time'] = time['date'].dt.time

print(time.head())

today - time['date']

# in seconds

print((today - time['date'])/np.timedelta64(1,'s'))

# in minutes

(today - time['date'])/np.timedelta64(1,'m')

# in hours

(today - time['date'])/np.timedelta64(1,'h')


  
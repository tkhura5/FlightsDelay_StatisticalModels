import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
import datetime
from datetime import *

#Reading the dataframe
df = pd.DataFrame()
df=pd.read_csv('flights.csv',low_memory=False)
df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].apply(pd.to_numeric)

group = pd.DataFrame()
group = df.groupby(['DATE']).mean()

df=df['DATE'].unique()
print df

#visually analysing time series using scatter plot
df['DATE'].unique().to_csv('date.csv',index=False)
x=df['DEPARTURE_TIME']
y=df['DEPARTURE_DELAY']
plt.scatter(x,y)
plt.show()

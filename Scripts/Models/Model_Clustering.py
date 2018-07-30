import pandas as pd
import numpy as np
import plotly
from plotly.graph_objs import *
import plotly.graph_objs as go

df_Flights = pd.pd.DataFrame()
df_Flights=pd.read_csv('flights.csv',low_memory=False)

df_Flights['DEPARTURE_DELAY'] = df_Flights['DEPARTURE_DELAY'].apply(pd.to_numeric)
group = df_Flights.groupby(['ORIGIN_AIRPORT','AIRLINE']).mean()

ST=df_Flights['SCHEDULED_TIME']
ET=df_Flights['ELAPSED_TIME']
DIS=df_Flights['DISTANCE']
print ST
print ET
print DIS

traceST=go.Scatter(
    x=ST,
    y=DIS
)
traceET=go.Scatter(
    x=ET,
    y=DIS
)
plotly.offline.plot ([traceST, traceET])

ORAIRPORTS = pd.DataFrame()
ORAIRPORTS = pd.read_csv('airports.csv')
dataframe = pd.DataFrame(data=ORAIRPORTS,columns=['airports'])
gb = dataframe.groupby(by=['airports'])
new=gb.size()

newcount=new.reset_index(name='times') 
print newcount

import pandas as pd
import numpy as np
import datetime
import csv
##import plotly as py
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
from numpy.linalg import *
from scipy.optimize import *
from pylab import *
##from mpl_toolkits.basemap import Basemap

#FLIGHTS
df_Flights = pd.DataFrame()
df_Flights = pd.DataFrame(columns=['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
                                   'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
                                   'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
df_Flights=pd.read_csv('flights.csv',low_memory=False)

#AIRPORTS
df_Airports = pd.DataFrame()
df_Airports = pd.DataFrame(columns=['IATA_CODE','AIRPORT','CITY','STATE','COUNTRY','LATITUDE','LONGITUDE'])   
df_Airports = pd.read_csv('airports.csv')

#AIRLINES
df_Airlines = pd.DataFrame()
df_Airlines = pd.DataFrame(columns=['IATA_CODE', 'AIRLINE_NAME'])   
df_Airlines = pd.read_csv('airlines.csv')

print 'Total Rows in df_Flights(before cleaning the data) ',df_Flights.shape[0]

##PREPARING THE DATA
##CLEANING the MAIN dataset ,that is, FLIGHTS(df_Flights)

    #Step 1-Combine columns to create a date
df_Flights['DATE'] = pd.to_datetime(df_Flights[['YEAR','MONTH', 'DAY']])

    #Step 2-Formatting various columns in time format
    #Function to convert columns to time datatype
##def format_hm(x):
##    if pd.isnull(x):
##        return np.nan
##    else:
##        if x == 2400: x = 0
##        x = "{0:04d}".format(int(x))
##        hm = datetime.time(int(x[0:2]), int(x[2:4]))
##        return hm
##
##df_Flights['SCHEDULED_DEPARTURE'] =df_Flights['SCHEDULED_DEPARTURE'].apply(format_hm)
##df_Flights['DEPARTURE_TIME'] = df_Flights['DEPARTURE_TIME'].apply(format_hm)
##df_Flights['SCHEDULED_ARRIVAL'] = df_Flights['SCHEDULED_ARRIVAL'].apply(format_hm)
##df_Flights['ARRIVAL_TIME'] = df_Flights['ARRIVAL_TIME'].apply(format_hm)

    #Step 3-Removing unwanted columns
df_Flights.drop(['YEAR','MONTH', 'DAY','DAY_OF_WEEK','TAIL_NUMBER','FLIGHT_NUMBER','WHEELS_ON','WHEELS_OFF','TAXI_IN','TAXI_OUT'], axis = 1, inplace = True)

    #Step 4-Deleting/Replacing blank value rows
df_Flights.dropna(subset=['SCHEDULED_DEPARTURE'],inplace = True)
df_Flights.dropna(subset=['DEPARTURE_TIME'],inplace = True)
df_Flights.dropna(subset=['SCHEDULED_ARRIVAL'],inplace = True)
df_Flights.dropna(subset=['ARRIVAL_TIME'],inplace = True)
df_Flights.DIVERTED.fillna(0, inplace=True)
df_Flights.CANCELLED.fillna(0, inplace=True)
df_Flights.CANCELLATION_REASON.fillna('', inplace=True)
df_Flights.SCHEDULED_TIME.fillna(0, inplace=True)
df_Flights.ELAPSED_TIME.fillna(0, inplace=True)
df_Flights.AIR_TIME.fillna(0, inplace=True)
df_Flights.AIR_SYSTEM_DELAY.fillna(0, inplace=True)
df_Flights.SECURITY_DELAY.fillna(0, inplace=True)
df_Flights.AIRLINE_DELAY.fillna(0, inplace=True)
df_Flights.LATE_AIRCRAFT_DELAY.fillna(0, inplace=True)
df_Flights.WEATHER_DELAY.fillna(0, inplace=True)

    #Step 5-Deleting/Replacing CANCELLED/DIVERTED flights as they won't affect delays
####Flights_cancelled=df_Flights.CANCELLED[df_Flights['CANCELLED']==1].count()
####Flights_diverted=df_Flights.DIVERTED[df_Flights['DIVERTED']==1].count()
####p_Flights_cancelled=(Flights_cancelled*1.0/Total_flights) * 100
####p_Flights_diverted=(Flights_diverted*1.0/Total_flights) * 100
####print 'Total Flights : ', Total_flights
####print 'Flights cancelled : ', Flights_cancelled
####print 'Percentage of flights cancelled : ',p_Flights_cancelled
####print ''
####print 'Total Flights : ', Total_flights
####print 'Flights diverted : ', Flights_diverted
####print 'Percentage of flights diverted : ',p_Flights_diverted
####print ''
df_Flights=df_Flights[df_Flights.CANCELLED == 0]
df_Flights=df_Flights[df_Flights.DIVERTED == 0]

    #Step 6-Rearranging data to make more sense
df_Flights=df_Flights[['DATE','AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                      'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
Total_flights=df_Flights.shape[0]
print 'Total Rows in df_Flights(after cleaning the data) ',df_Flights.shape[0]

#MERGING to the MAIN dataset ,that is, FLIGHTS(df_Flights)
df_Flights=pd.merge(df_Flights, df_Airlines, left_on='AIRLINE',right_on='IATA_CODE', how='inner')
df_Flights.drop(['IATA_CODE'], axis = 1, inplace = True)
df_Flights=df_Flights[['DATE','AIRLINE','AIRLINE_NAME', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                      'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]


df_Flights1 = df_Flights[['DEPARTURE_DELAY','ARRIVAL_DELAY','DEPARTURE_TIME', 'SCHEDULED_TIME', 'ARRIVAL_TIME', 'AIR_SYSTEM_DELAY',
                          'SECURITY_DELAY', 'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean(),
            'median':group.median(), 'variance':group.var(),
            'standard deviation':group.std(), 'skew':group.skew(),
            'kurtosis':group.kurt()}

##print get_stats(df_Flights1)
print
corr = df_Flights1.corr()
print corr 
print
##print df_Flights1.cov()

#Univariate plots
#Histogram
df_Flights.hist()
plt.show()

df_Flights.boxplot(column="DEPARTURE_DELAY", by="AIRLINE", figsize= (14,14))

#Correlation matrix
plt.title('Airline Delay Correlation')
names=['DEPARTURE_DELAY','ARRIVAL_DELAY','DEPARTURE_TIME','SCHEDULED_TIME','ARRIVAL_TIME','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
ax.tick_params(axis='x', rotation=45)
plt.show()

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Airline Delay Correlation')
    labels=['DEPARTURE_DELAY','ARRIVAL_DELAY','DEPARTURE_TIME','SCHEDULED_TIME','ARRIVAL_TIME','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.7,.75,.8,.85,.9,.95,1])
    ax1.tick_params(axis='x', rotation=45)
    plt.show()

correlation_matrix(df_Flights1)
x = np.array(df_Flights['DEPARTURE_DELAY'])
y = np.array(df_Flights['AIRLINE_DELAY'])
slope,intercept,r_value,p_value,slope_std_error = stats.linregress(x,y)
y_modeled = x*slope+intercept
plot(x,y,'ob',markersize=2)
plot(x,y_modeled,'-r',linewidth=1)
title('Linear Regression Fit Plot')
show() 

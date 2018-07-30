import pandas as pd
import numpy as np
import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
from numpy.linalg import *
from scipy.optimize import *
from pylab import *
from mpl_toolkits.basemap import Basemap
import plotly
from plotly.graph_objs import *

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
def format_hm(x):
    if pd.isnull(x):
        return np.nan
    else:
        if x == 2400: x = 0
        x = "{0:04d}".format(int(x))
        hm = datetime.time(int(x[0:2]), int(x[2:4]))
        return hm

df_Flights['SCHEDULED_DEPARTURE'] =df_Flights['SCHEDULED_DEPARTURE'].apply(format_hm)
df_Flights['DEPARTURE_TIME'] = df_Flights['DEPARTURE_TIME'].apply(format_hm)
df_Flights['SCHEDULED_ARRIVAL'] = df_Flights['SCHEDULED_ARRIVAL'].apply(format_hm)
df_Flights['ARRIVAL_TIME'] = df_Flights['ARRIVAL_TIME'].apply(format_hm)

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
Flights_cancelled=df_Flights.CANCELLED[df_Flights['CANCELLED']==1].count()
Flights_diverted=df_Flights.DIVERTED[df_Flights['DIVERTED']==1].count()
p_Flights_cancelled=(Flights_cancelled*1.0/Total_flights) * 100
p_Flights_diverted=(Flights_diverted*1.0/Total_flights) * 100
print 'Total Flights : ', Total_flights
print 'Flights cancelled : ', Flights_cancelled
print 'Percentage of flights cancelled : ',p_Flights_cancelled
print ''
print 'Total Flights : ', Total_flights
print 'Flights diverted : ', Flights_diverted
print 'Percentage of flights diverted : ',p_Flights_diverted
print ''
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

#DESCRIPTIVE ANALYSIS
#CATEGORICAL VARIABLES
#AIRLINES
#Total number of flights of each airline
num_of_FLIGHTS = dict(df_Flights['AIRLINE'].value_counts())
print 'airlines with the number of FLIGHTS ',num_of_FLIGHTS
max_num_of_FLIGHTS = max(num_of_FLIGHTS, key=num_of_FLIGHTS.get)  
print 'airline with the maximum number of FLIGHTS',(max_num_of_FLIGHTS, num_of_FLIGHTS[max_num_of_FLIGHTS])
#Plotting the total number of flights of each airline - Pie chart
labels = 'Southwest Airlines', 'Delta Air Lines', 'American Airlines', 'Skywest Airlines','Atlantic Southeast Airlines','United Air Lines','American Eagle Airlines','JetBlue Airways','US Airways','Alaska Airlines','Spirit Air Lines','Frontier Airlines','Hawaiian Airlines','Virgin America'
sizes = [num_of_FLIGHTS['WN'], num_of_FLIGHTS['DL'], num_of_FLIGHTS['AA'], num_of_FLIGHTS['OO'], num_of_FLIGHTS['EV'], num_of_FLIGHTS['UA'], num_of_FLIGHTS['MQ'], num_of_FLIGHTS['B6'], num_of_FLIGHTS['US'],
         num_of_FLIGHTS['AS'], num_of_FLIGHTS['NK'], num_of_FLIGHTS['F9'], num_of_FLIGHTS['HA'], num_of_FLIGHTS['VX']]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','seagreen','red','orange','green','purple','pink','lightsalmon','lightgrey','cyan','magenta']
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # Explode Southwest Airlines Co. as it has max no. of flights. 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#From data we find, Southwest Airlines Co. has the maximum flight count in USA.
#Let us find if Southwest Airlines Co. also has the maximum average delay.
#Calculating the total average delay of each airline
df_Airline_Delay = pd.DataFrame()
df_Airline_Delay =df_Flights.groupby(['AIRLINE'], as_index=False)[["DEPARTURE_DELAY", "ARRIVAL_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","AIRLINE"]].sum()
df_Airline_Delay['COUNT']= df_Airline_Delay['AIRLINE'].map(df_Flights['AIRLINE'].value_counts())
df_Airline_Delay['TOTAL_DELAY'] = df_Airline_Delay['DEPARTURE_DELAY'] + df_Airline_Delay['ARRIVAL_DELAY'] + df_Airline_Delay['AIR_SYSTEM_DELAY'] + df_Airline_Delay['SECURITY_DELAY'] + df_Airline_Delay['AIRLINE_DELAY'] +df_Airline_Delay['LATE_AIRCRAFT_DELAY'] + df_Airline_Delay['WEATHER_DELAY']
df_Airline_Delay['AVG_DELAY'] = df_Airline_Delay['TOTAL_DELAY']/df_Airline_Delay['COUNT']
print 'total average delay of each airline', df_Airline_Delay
print 'airline with the maximum average delay',df_Airline_Delay.loc[df_Airline_Delay['AVG_DELAY'].idxmax()]
#Plotting the delays of each airline - Pie chart
labels = 'Southwest Airlines', 'Delta Air Lines', 'American Airlines', 'Skywest Airlines','Atlantic Southeast Airlines','United Air Lines','American Eagle Airlines','JetBlue Airways','US Airways','Alaska Airlines','Spirit Air Lines','Frontier Airlines','Hawaiian Airlines','Virgin America'
sizes = [df_Airline_Delay['AVG_DELAY'][13], df_Airline_Delay['AVG_DELAY'][3], df_Airline_Delay['AVG_DELAY'][0], df_Airline_Delay['AVG_DELAY'][9],
         df_Airline_Delay['AVG_DELAY'][4], df_Airline_Delay['AVG_DELAY'][10], df_Airline_Delay['AVG_DELAY'][7], df_Airline_Delay['AVG_DELAY'][2],
         df_Airline_Delay['AVG_DELAY'][11], df_Airline_Delay['AVG_DELAY'][1], df_Airline_Delay['AVG_DELAY'][8], df_Airline_Delay['AVG_DELAY'][5],
         df_Airline_Delay['AVG_DELAY'][6], df_Airline_Delay['AVG_DELAY'][12]]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','seagreen','red','orange','green','purple','pink','lightsalmon','lightgrey','cyan','magenta']
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0)  # Explode spirit airlines as it has max mean delay. 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
#Plotting the delays of each airline - Bar plot
objects = ('Southwest', 'Delta', 'American', 'Skywest','Atlantic Southeast','United','American Eagle','JetBlue','US','Alaska','Spirit','Frontier','Hawaiian','Virgin')
y_pos = np.arange(len(objects))
performance = [df_Airline_Delay['AVG_DELAY'][13], df_Airline_Delay['AVG_DELAY'][3], df_Airline_Delay['AVG_DELAY'][0], df_Airline_Delay['AVG_DELAY'][9],
               df_Airline_Delay['AVG_DELAY'][4], df_Airline_Delay['AVG_DELAY'][10], df_Airline_Delay['AVG_DELAY'][7], df_Airline_Delay['AVG_DELAY'][2],
               df_Airline_Delay['AVG_DELAY'][11], df_Airline_Delay['AVG_DELAY'][1], df_Airline_Delay['AVG_DELAY'][8], df_Airline_Delay['AVG_DELAY'][5],
               df_Airline_Delay['AVG_DELAY'][6], df_Airline_Delay['AVG_DELAY'][12]]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xticks(rotation=20)
plt.ylabel('Delay')
plt.xlabel('Airlines')
plt.title('Average Delay Airlines') 
plt.show()
#After plotting we find out that Southwest Airlines Co. may have the maximum number of flights running is the USA but it doesnt account for the maximum average delay.
#We Noticed that the Frontier airlines has the maximum average delay. 
#Now lets break this delay in two parts, Departure delay and Arrival delay to analyze them separately.

#Departure Delay per airline
df_Dep_Delay = pd.DataFrame()
df_Dep_Delay =df_Flights.groupby(['AIRLINE'], as_index=False)[["DEPARTURE_DELAY", "ARRIVAL_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","AIRLINE"]].sum()
df_Dep_Delay['COUNT']= df_Dep_Delay['AIRLINE'].map(df_Flights['AIRLINE'].value_counts())
df_Dep_Delay['AVG_DEPT_DELAY'] = df_Dep_Delay['DEPARTURE_DELAY']/df_Dep_Delay['COUNT']
print 'total average departure delay of each airline', df_Dep_Delay
print 'airline with the maximum average departure delay',df_Dep_Delay.loc[df_Dep_Delay['AVG_DEPT_DELAY'].idxmax()]
#Plotting airlines on the basis of average DEPARTURE delay
objects = ('Southwest', 'Delta', 'American', 'Skywest','Atlantic Southeast','United','American Eagle','JetBlue','US','Alaska','Spirit','Frontier','Hawaiian','Virgin')
y_pos = np.arange(len(objects))
performance = [df_Dep_Delay['AVG_DEPT_DELAY'][13], df_Dep_Delay['AVG_DEPT_DELAY'][3], df_Dep_Delay['AVG_DEPT_DELAY'][0], df_Dep_Delay['AVG_DEPT_DELAY'][9], df_Dep_Delay['AVG_DEPT_DELAY'][4], df_Dep_Delay['AVG_DEPT_DELAY'][10],
df_Dep_Delay['AVG_DEPT_DELAY'][7], df_Dep_Delay['AVG_DEPT_DELAY'][2], df_Dep_Delay['AVG_DEPT_DELAY'][11], df_Dep_Delay['AVG_DEPT_DELAY'][1], df_Dep_Delay['AVG_DEPT_DELAY'][8],
df_Dep_Delay['AVG_DEPT_DELAY'][5], df_Dep_Delay['AVG_DEPT_DELAY'][6], df_Dep_Delay['AVG_DEPT_DELAY'][12]]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xticks(rotation=20)
plt.ylabel('Departure Delay')
plt.xlabel('Airlines')
plt.title('Average Departure Delay Airlines') 
plt.show()

#Arrival Delay per airline
df_Arr_Delay = pd.DataFrame()
df_Arr_Delay =df_Flights.groupby(['AIRLINE'], as_index=False)[["DEPARTURE_DELAY", "ARRIVAL_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","AIRLINE"]].sum()
df_Arr_Delay['COUNT']= df_Arr_Delay['AIRLINE'].map(df_Flights['AIRLINE'].value_counts())
df_Arr_Delay['AVG_ARR_DELAY'] = df_Arr_Delay['ARRIVAL_DELAY']/df_Arr_Delay['COUNT']
print 'total average departure delay of each airline', df_Arr_Delay
print 'airline with the maximum average departure delay',df_Arr_Delay.loc[df_Arr_Delay['AVG_ARR_DELAY'].idxmax()]
#Plotting Flights with mean departure delay
objects = ('Southwest', 'Delta', 'American', 'Skywest','Atlantic Southeast','United','American Eagle','JetBlue','US','Alaska','Spirit','Frontier','Hawaiian','Virgin')
y_pos = np.arange(len(objects))
performance = [df_Arr_Delay['AVG_ARR_DELAY'][13], df_Arr_Delay['AVG_ARR_DELAY'][3], df_Arr_Delay['AVG_ARR_DELAY'][0], df_Arr_Delay['AVG_ARR_DELAY'][9], df_Arr_Delay['AVG_ARR_DELAY'][4], df_Arr_Delay['AVG_ARR_DELAY'][10],
df_Arr_Delay['AVG_ARR_DELAY'][7], df_Arr_Delay['AVG_ARR_DELAY'][2], df_Arr_Delay['AVG_ARR_DELAY'][11], df_Arr_Delay['AVG_ARR_DELAY'][1], df_Arr_Delay['AVG_ARR_DELAY'][8],
df_Arr_Delay['AVG_ARR_DELAY'][5], df_Arr_Delay['AVG_ARR_DELAY'][6], df_Arr_Delay['AVG_ARR_DELAY'][12]]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xticks(rotation=20)
plt.ylabel('Arrival Delay')
plt.xlabel('Airlines')
plt.title('Average Arrival Delay Airlines') 
plt.show()

#After looking at the total flights number pie chart , we conclude that Southwest Airlines Co. has the maximum number of flights flying in the year 2015 but does not play much of a role in the delays.
#Also, we conclude that although Spirit(2%),Frontier(1.6%) and United(8.8%) Airlines hold very less share in the same plot ,they are the main contributors in the delays for the year 2015. 

#Here is some information about the airlines causing the most delays.
#The flights with the maximum average delay are Spirit,Frontier and United Airlines respectively.
                 #the maximum DEPARTURE delay are Spirit,United and Frontier Airlines respectively.
                 #the maximum ARRIVAL delay are Spirit,Frontier and Jetblue Airlines respectively.


#LETS SHIFT OUR ATTENTION TO THE AIRPORTS NOW
#AIRPORTS
#Origin Airport with the maximum flights load
origin_airport_num_of_FLIGHTS = dict(df_Flights['ORIGIN_AIRPORT'].value_counts())
print origin_airport_num_of_FLIGHTS
busiest_origin = max(origin_airport_num_of_FLIGHTS, key=origin_airport_num_of_FLIGHTS.get)  
print 'Busiest Origin Airport ',(busiest_origin, origin_airport_num_of_FLIGHTS[busiest_origin])

#Destination Airport with the maximum flights load
dest_airport_num_of_FLIGHTS = dict(df_Flights['DESTINATION_AIRPORT'].value_counts())
print dest_airport_num_of_FLIGHTS
busiest_destination = max(dest_airport_num_of_FLIGHTS, key=dest_airport_num_of_FLIGHTS.get)  
print 'Busiest Destination Airport ',(busiest_destination, dest_airport_num_of_FLIGHTS[busiest_destination])

#MERGING to two new dataset
df_Flights_Origin=pd.merge(df_Flights, df_Airports, left_on='ORIGIN_AIRPORT',right_on='IATA_CODE', how='inner')
df_Flights_Origin.drop(['IATA_CODE'], axis = 1, inplace = True)
df_Flights_Origin=df_Flights_Origin[['DATE','AIRLINE','AIRLINE_NAME','AIRPORT','CITY','STATE','COUNTRY','LATITUDE','LONGITUDE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                      'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]

df_Flights_Destination=pd.merge(df_Flights, df_Airports, left_on='DESTINATION_AIRPORT',right_on='IATA_CODE', how='inner')
df_Flights_Destination.drop(['IATA_CODE'], axis = 1, inplace = True)
df_Flights_Destination=df_Flights_Destination[['DATE','AIRLINE','AIRLINE_NAME','AIRPORT','CITY','STATE','COUNTRY','LATITUDE','LONGITUDE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                      'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]

##Delay per ORIGIN airport
df_Delay_ORIGIN_airport = pd.DataFrame()
df_Delay_ORIGIN_airport =df_Flights_Origin.groupby(['AIRPORT'], as_index=False)[["ORIGIN_AIRPORT","AIRPORT","DEPARTURE_DELAY", "ARRIVAL_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY"]].sum()
df_Delay_ORIGIN_airport['COUNT']= df_Delay_ORIGIN_airport['AIRPORT'].map(df_Flights_Origin['AIRPORT'].value_counts())
df_Delay_ORIGIN_airport['AVG_DEPT_DELAY'] = df_Delay_ORIGIN_airport['DEPARTURE_DELAY']/df_Delay_ORIGIN_airport['COUNT']
df_Delay_ORIGIN_airport['AVG_ARR_DELAY'] = df_Delay_ORIGIN_airport['ARRIVAL_DELAY']/df_Delay_ORIGIN_airport['COUNT']
df_Delay_ORIGIN_airport['AVG_ARR_DEPT_DELAY'] = df_Delay_ORIGIN_airport['AVG_DEPT_DELAY'] + df_Delay_ORIGIN_airport['AVG_ARR_DELAY']
df_Delay_ORIGIN_airport['TOTAL_DELAY'] = df_Delay_ORIGIN_airport['DEPARTURE_DELAY'] + df_Delay_ORIGIN_airport['ARRIVAL_DELAY'] + df_Delay_ORIGIN_airport['AIR_SYSTEM_DELAY'] + df_Delay_ORIGIN_airport['SECURITY_DELAY'] + df_Delay_ORIGIN_airport['AIRLINE_DELAY'] + df_Delay_ORIGIN_airport['LATE_AIRCRAFT_DELAY'] + df_Delay_ORIGIN_airport['WEATHER_DELAY']
df_Delay_ORIGIN_airport['AVG_TOTAL_DELAY'] = df_Delay_ORIGIN_airport['TOTAL_DELAY']/df_Delay_ORIGIN_airport['COUNT']
#print df_Delay_ORIGIN_airport

print 'avg depature delay'
print df_Delay_ORIGIN_airport[df_Delay_ORIGIN_airport['AVG_DEPT_DELAY'] == df_Delay_ORIGIN_airport['AVG_DEPT_DELAY'].max()]
print 'avg arrival delay'
print df_Delay_ORIGIN_airport[df_Delay_ORIGIN_airport['AVG_ARR_DELAY'] == df_Delay_ORIGIN_airport['AVG_ARR_DELAY'].max()]
print 'avg total arrival departure delay'
print df_Delay_ORIGIN_airport[df_Delay_ORIGIN_airport['AVG_ARR_DEPT_DELAY'] == df_Delay_ORIGIN_airport['AVG_ARR_DEPT_DELAY'].max()]
print 'avg total delay'
print df_Delay_ORIGIN_airport[df_Delay_ORIGIN_airport['AVG_TOTAL_DELAY'] == df_Delay_ORIGIN_airport['AVG_TOTAL_DELAY'].max()]
print df_Delay_ORIGIN_airport.loc[df_Delay_ORIGIN_airport['AVG_TOTAL_DELAY'].idxmax()]

##Delay per DESTINATION airport
df_Delay_DESTINATION_airport = pd.DataFrame()
df_Delay_DESTINATION_airport =df_Flights_Destination.groupby(['AIRPORT'], as_index=False)[["DESTINATION_AIRPORT","AIRPORT","DEPARTURE_DELAY", "ARRIVAL_DELAY", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY"]].sum()
df_Delay_DESTINATION_airport['COUNT']= df_Delay_DESTINATION_airport['AIRPORT'].map(df_Flights_Destination['AIRPORT'].value_counts())
df_Delay_DESTINATION_airport['AVG_DEPT_DELAY'] = df_Delay_DESTINATION_airport['DEPARTURE_DELAY']/df_Delay_DESTINATION_airport['COUNT']
df_Delay_DESTINATION_airport['AVG_ARR_DELAY'] = df_Delay_DESTINATION_airport['ARRIVAL_DELAY']/df_Delay_DESTINATION_airport['COUNT']
df_Delay_DESTINATION_airport['AVG_ARR_DEPT_DELAY'] = df_Delay_DESTINATION_airport['AVG_DEPT_DELAY'] + df_Delay_DESTINATION_airport['AVG_ARR_DELAY']
df_Delay_DESTINATION_airport['TOTAL_DELAY'] = df_Delay_DESTINATION_airport['DEPARTURE_DELAY'] + df_Delay_DESTINATION_airport['ARRIVAL_DELAY'] + df_Delay_DESTINATION_airport['AIR_SYSTEM_DELAY'] + df_Delay_DESTINATION_airport['SECURITY_DELAY'] + df_Delay_DESTINATION_airport['AIRLINE_DELAY'] + df_Delay_DESTINATION_airport['LATE_AIRCRAFT_DELAY'] + df_Delay_DESTINATION_airport['WEATHER_DELAY']
df_Delay_DESTINATION_airport['AVG_TOTAL_DELAY'] = df_Delay_DESTINATION_airport['TOTAL_DELAY']/df_Delay_DESTINATION_airport['COUNT']
#print df_Delay_DESTINATION_airport

print 'avg dept delay'
print df_Delay_DESTINATION_airport[df_Delay_DESTINATION_airport['AVG_DEPT_DELAY'] == df_Delay_DESTINATION_airport['AVG_DEPT_DELAY'].max()]
print 'avg arrival delay'
print df_Delay_DESTINATION_airport[df_Delay_DESTINATION_airport['AVG_ARR_DELAY'] == df_Delay_DESTINATION_airport['AVG_ARR_DELAY'].max()]
print 'avg total arrival departure delay'
print df_Delay_DESTINATION_airport[df_Delay_DESTINATION_airport['AVG_ARR_DEPT_DELAY'] == df_Delay_DESTINATION_airport['AVG_ARR_DEPT_DELAY'].max()]
print 'avg total delay'
print df_Delay_DESTINATION_airport[df_Delay_DESTINATION_airport['AVG_TOTAL_DELAY'] == df_Delay_DESTINATION_airport['AVG_TOTAL_DELAY'].max()]
print df_Delay_DESTINATION_airport.loc[df_Delay_DESTINATION_airport['AVG_TOTAL_DELAY'].idxmax()]

#After analyzing we observed that the busiest origin and destination airport is Atlanta airport.
#With further analysis we find that the maximum average delay at the origin airport and destination airport is on Wilmington airport but not Atlanta airport.
#Hence, the busiest airport doesnot account for the maximum average delay.
#Results -For Origin airport
            # avg depature delay - Wilmington Airport
            # avg arrival delay - Wilmington Airport
            # avg total arrival and departure delay - Wilmington Airport
            # avg total delay - Wilmington Airport
#Results -For Destination airport
            # avg depature delay - Guam International Airport
            # avg arrival delay - St. Cloud Regional Airport
            # avg total arrival and departure delay - Wilmington Airport
            # avg total delay - Wilmington Airport

#MERGING to two new dataset
df_Flights=pd.merge(df_Flights, df_Airports, left_on='ORIGIN_AIRPORT',right_on='IATA_CODE', how='inner')
df_Flights.drop(['IATA_CODE'], axis = 1, inplace = True)
df_Flights=df_Flights[['DATE','AIRLINE','AIRLINE_NAME','AIRPORT','CITY','STATE','COUNTRY','LATITUDE','LONGITUDE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                      'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]

#Plotting all the airports on a map of USA
mapbox_access_token = 'pk.eyJ1IjoiZW5ncmFtYXIiLCJhIjoiY2o4YXpqNjlnMDhsYTJ3cDYyenVrY2FtNCJ9.XTFWmcvOXipEb9Fc7oxkWg'

df = pd.read_csv('airports.csv')
site_lat = df.LATITUDE
site_lon = df.LONGITUDE
locations_name = df.AIRPORT

data = Data([
    Scattermapbox(
        lat=site_lat,
        lon=site_lon,
        mode='markers',
        marker=Marker(
            size=10,
            color='rgb(255, 87, 51)',
            opacity=1.0
        ),
        text=locations_name,
        hoverinfo='AIRPORTS IN USA'
    )
]
)
        
layout = Layout(
    title='Airports',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=38,
            lon=-94
        ),
        pitch=0,
        zoom=3,
        style='light'
    ),
)

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename='airports.html')

##filename = 'airports.csv'
##lats, lons = [], []
##with open(filename) as f:   
##    reader = csv.reader(f)    
##    next(reader)   
##    for row in reader:
##        lats.append(float(row[5]))
##        lons.append(float(row[6]))
##
##themap = Basemap(projection='gall',
##                 llcrnrlon=-180,
##                 urcrnrlon=-50,
##                 llcrnrlat=10,
##                 urcrnrlat=75,      
##                 resolution = 'l',
##                 area_thresh = 100000.0,
##              )
##
##themap.drawcoastlines()
##themap.drawcountries()
##themap.fillcontinents(color = 'gainsboro')
##themap.drawmapboundary(fill_color='steelblue')
##
##x, y = themap(lons,lats)
##themap.plot(x, y, 'ro', color='red',markersize=4)
##
##plt.show()


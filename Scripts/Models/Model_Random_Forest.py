import numpy as np
import pandas as pd
import csv
import os
import tabulate
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, roc_curve
import warnings
np.random.seed(10)

#1)Load data:Flights which fly in December(December.csv)
df_Flights_Dec = pd.DataFrame(columns=['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT',
                                       'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
                                       'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
df_Flights_Dec=pd.read_csv('E:/flight-delays/flights.csv',low_memory=False)
df_Flights_Dec=df_Flights_Dec[['YEAR','MONTH', 'DAY','DAY_OF_WEEK','AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                               'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
Total_Flights=df_Flights_Dec.shape[0]
print 'TOTAL FLIGHTS ',Total_Flights
print df_Flights_Dec.head()
df_Flights_Dec['TEMP'] = np.random.uniform(0, 1, len(df_Flights_Dec)) <= .8
train, test = df_Flights_Dec[df_Flights_Dec['TEMP']==True], df_Flights_Dec[df_Flights_Dec['TEMP']==False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
df_Flights_Dec['ARRIVAL_DELAY_BINARY'] = np.where(df_Flights_Dec['ARRIVAL_DELAY']>=0, 1, 0)
fdata = df_Flights_Dec;

#2)Deleting/Replacing blank value rows
df_Flights_Dec.dropna(subset=['SCHEDULED_DEPARTURE'],inplace = True)
df_Flights_Dec.dropna(subset=['DEPARTURE_TIME'],inplace = True)
df_Flights_Dec.dropna(subset=['SCHEDULED_ARRIVAL'],inplace = True)
df_Flights_Dec.dropna(subset=['ARRIVAL_TIME'],inplace = True)
df_Flights_Dec.DIVERTED.fillna(0, inplace=True)
df_Flights_Dec.CANCELLED.fillna(0, inplace=True)
df_Flights_Dec.CANCELLATION_REASON.fillna('', inplace=True)
df_Flights_Dec.SCHEDULED_TIME.fillna(0, inplace=True)
df_Flights_Dec.ELAPSED_TIME.fillna(0, inplace=True)
df_Flights_Dec.AIR_TIME.fillna(0, inplace=True)
df_Flights_Dec.AIR_SYSTEM_DELAY.fillna(0, inplace=True)
df_Flights_Dec.SECURITY_DELAY.fillna(0, inplace=True)
df_Flights_Dec.AIRLINE_DELAY.fillna(0, inplace=True)
df_Flights_Dec.LATE_AIRCRAFT_DELAY.fillna(0, inplace=True)
df_Flights_Dec.WEATHER_DELAY.fillna(0, inplace=True)

#3)Normalizing the data
classDistribution = fdata['ARRIVAL_DELAY_BINARY'].value_counts()
print('Class imbalance:')
print(classDistribution)
fdata = fdata.sample(frac=1).reset_index(drop=True)
zero = fdata[fdata['ARRIVAL_DELAY_BINARY'] == 0].tail(classDistribution.min())
one = fdata[fdata['ARRIVAL_DELAY_BINARY'] == 1]
data = zero.append(one) 
del zero, one
data = data.sample(frac=1).reset_index(drop=True)
print('Class imbalance evened out:')
print(data['ARRIVAL_DELAY_BINARY'].value_counts())
print (len(data.columns))

#4)Rearranging data to make more sense
df_Flights_Dec=df_Flights_Dec[['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK','AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT','SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY','SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                      'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME','DISTANCE','DIVERTED', 'CANCELLED', 'CANCELLATION_REASON','AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
Total_flights=df_Flights_Dec.shape[0]
print 'Total Rows in df_Flights(after cleaning the data) ',df_Flights_Dec.shape[0]

#5)Exploratory analysis
avgLate = np.sum(data['ARRIVAL_DELAY_BINARY'])/len(data['ARRIVAL_DELAY_BINARY'])
attributes = ['DAY','DAY_OF_WEEK','AIRLINE','DISTANCE']
for i,pred in enumerate(attributes):
    group = data.groupby([pred], as_index=False).aggregate(np.mean)[[pred, 'ARRIVAL_DELAY_BINARY']]
    group.sort_values(by=pred, inplace=True)    
    group.plot.bar(x=pred, y='ARRIVAL_DELAY_BINARY')
    plt.axhline(y=avgLate, label='average')
    plt.ylabel('Percent of FLIGHTS that arrive late')
    plt.title(pred)
    plt.legend().remove()
    plt.show()

#6)Label encoding:Conversion of categorical values to numericals
le = LabelEncoder()
data["AIRLINE"] = le.fit_transform(data["AIRLINE"])
AIRLINE = list(le.classes_)
data["ORIGIN_AIRPORT"] = le.fit_transform(data["ORIGIN_AIRPORT"])
ORIGIN_AIRPORT = list(le.classes_)
data["DESTINATION_AIRPORT"] = le.fit_transform(data["DESTINATION_AIRPORT"])
DESTINATION_AIRPORT = list(le.classes_)
data["DEPARTURE_TIME"] = le.fit_transform(data["DEPARTURE_TIME"])
DEPARTURE_TIME = list(le.classes_)
data["DEPARTURE_DELAY"] = le.fit_transform(data["DEPARTURE_DELAY"])
DEPARTURE_DELAY = list(le.classes_)
data["ARRIVAL_TIME"] = le.fit_transform(data["ARRIVAL_TIME"])
ARRIVAL_TIME = list(le.classes_)
data["ARRIVAL_DELAY"] = le.fit_transform(data["ARRIVAL_DELAY"])
ARRIVAL_DELAY = list(le.classes_)
data["SCHEDULED_TIME"] = le.fit_transform(data["SCHEDULED_TIME"])
SCHEDULED_TIME = list(le.classes_)
data["ELAPSED_TIME"] = le.fit_transform(data["ELAPSED_TIME"])
DEPARTURE_DELAY = list(le.classes_)
data["AIR_TIME"] = le.fit_transform(data["AIR_TIME"])
AIR_TIME = list(le.classes_)
data["CANCELLATION_REASON"] = le.fit_transform(data["CANCELLATION_REASON"])
CANCELLATION_REASON = list(le.classes_)
data["AIR_SYSTEM_DELAY"] = le.fit_transform(data["AIR_SYSTEM_DELAY"])
AIR_SYSTEM_DELAY = list(le.classes_)
data["SECURITY_DELAY"] = le.fit_transform(data["SECURITY_DELAY"])
SECURITY_DELAY = list(le.classes_)
data["AIRLINE_DELAY"] = le.fit_transform(data["AIRLINE_DELAY"])
AIRLINE_DELAY = list(le.classes_)
data["LATE_AIRCRAFT_DELAY"] = le.fit_transform(data["LATE_AIRCRAFT_DELAY"])
LATE_AIRCRAFT_DELAY = list(le.classes_)
data["WEATHER_DELAY"] = le.fit_transform(data["WEATHER_DELAY"])
WEATHER_DELAY = list(le.classes_)
data["TEMP"] = le.fit_transform(data["TEMP"])
TEMP = list(le.classes_)
print data.dtypes
print data.describe()

#RANDOM FOREST
#Random Forest is chosen for due to following reasons:
#(1) Medium data size
#(2) Categorical nature of many variables
#(3) Non-linearity of dataset (LinearRegression was attempted but had high MSE; Linear relation doesn't exist for many variables)
#(4) Underlying complex dependencies between variables
#(5) Non-parameteric approach without much assumptions

#Creation of dataset for modeling
Delay_YesNo = data['ARRIVAL_DELAY_BINARY']
data.drop(['ARRIVAL_DELAY_BINARY'], axis=1, inplace=True)
data_part2 = pd.DataFrame(data)
#Train/test split:Train:Test split = 80:20
X_train, X_test, Y_train, Y_test = train_test_split(data_part2, Delay_YesNo, test_size=0.2, random_state=0)
startTimeGS = datetime.now()
rf = RandomForestClassifier()
param_grid = {
                 'n_estimators': [5],#10-5
                 #'min_samples_split': [2, 4],
                 #'min_samples_leaf': [2, 4],
                 #'max_features': ['sqrt', 'log2'],
                 "criterion" : ["gini"]
             }
grid_rf = GridSearchCV(rf, param_grid, cv=3)#cv=3-5
grid_rf.fit(X_train, Y_train)
bestModel = grid_rf.best_estimator_
bestParameters = grid_rf.best_params_
gridScores = grid_rf.grid_scores_
print('Random forest Grid Search with non-redundant variables took [', datetime.now() - startTimeGS, '] seconds.')
#Best model, corresponding parameters and CV results
print(bestModel)
print(bestParameters)
print gridScores
#Model selection
#Best model and parameters from above are used to train the final model on entire training set
#3-fold Cross validation is performed to find the overall error
startTimeRF = datetime.now()
rf = RandomForestClassifier(n_estimators = bestParameters.get('n_estimators'), 
#                           min_samples_split=bestParameters.get('min_samples_split'),
#                           min_samples_leaf = bestParameters.get('min_samples_leaf'),
#                           max_features = bestParameters.get('max_features'),
                           criterion = bestParameters.get('criterion'))
cv = cross_validation.KFold(len(X_train), n_folds=10, shuffle=True, random_state=0)#改fold10
cvScores = cross_val_score(rf, X_train, Y_train, cv=cv)
print ('Mean cross validation score is: ' + str(np.mean(cvScores)))
rf.fit(X_train, Y_train)
print('Random forest training and testing with with non-redundant variables took [',datetime.now() - startTimeRF, '] seconds.')

#Performance evaluation
#Prediction is done on the hold out test set to evaluate performance
#Confusion matrix, Accuracy and Recall are computed
#ROC curve is also plotted to pictorically depict 'Area under curve (AUC)' as 'Accuracy'
Y_rf_pred = rf.predict(X_test)
labels = [0, 1]
cm = confusion_matrix(Y_test, Y_rf_pred,labels)
print('Accuracy: ' + str(np.round(100*float(cm[0][0]+cm[1][1])/float((cm[0][0]+cm[1][1] + cm[1][0] + cm[0][1])),2))+'%')
print('Recall: ' + str(np.round(100*float((cm[1][1]))/float((cm[1][0]+cm[1][1])),2))+'%')
print('Confusion matrix:')
print(cm)
fpr, tpr, _ = roc_curve(Y_test, Y_rf_pred)
auc = np.trapz(fpr,tpr)
print('Area under the ROC curve: ' + str(auc))
fig = plt.figure(1)
plt.plot(fpr,tpr,color='green')
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.title('Receiver operating characteristic (ROC)')
fig = plt.figure(2)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix for Random Forest classifier with original data')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

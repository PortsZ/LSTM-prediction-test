import pandas as pd
import nasdaqdatalink
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

nasdaqdatalink.read_key(filename="./.key") #loads api key

data = nasdaqdatalink.get_table('QDL/BITFINEX' , code='BTCUSD') #get the data

data = data.sort_values('date') #sort by date



#=================================================================================
data['HL_PCT'] = (data['high']-data['last'])/data['last'] * 100.0
data['PCT_daily_change'] = data['last'].pct_change() * 100
data['Dollar_Volume'] = data['volume'] * data['last']
#this is where we make money.
#=================================================================================

data = data[['date', 'last', 'HL_PCT', 'PCT_daily_change', 'Dollar_Volume']]

# Convert 'date' to datetime and set as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
# print(data)

#last stands for BTC "close" price of the day and it is what we want to predict
forecast_col = 'last' #the col we want to forecast
data.fillna(-99999, inplace=True) #fill nan with -99999

forecast_out = int(math.ceil(0.003*len(data))) #how many days we want to predict
#Tries to predict 11 days of the dataframe (we got ~3450 days in this dataset)
original_data = data[['last']].copy()

data['label'] = data[forecast_col].shift(-forecast_out) 
#this is what we're trying to predict, 1% into the future using the shift

X = np.array(data.drop('label', axis=1)) #remove label from dataset
X = preprocessing.scale(X)
X = X[:-forecast_out] #removes "forecast_out" elements (11 in this case) from the end of the array 
X_lately = X[-forecast_out:] #what we're going to predict against

data.dropna(inplace=True)
Y = np.array(data['label']) #labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

# clf=LinearRegression(n_jobs=-1)
# # clf=svm.SVR(kernel='sigmoid')
# clf.fit(X_train,Y_train)


# with open('linearregression.pickle', 'wb') as f: # saves into a file
#     pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle', 'rb') #load the file
clf = pickle.load(pickle_in) 


accuracy = clf.score(X_test,Y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set: #this is to get the dates in the axis
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    
original_data['last'].plot()
data['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
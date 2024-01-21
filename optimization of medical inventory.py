
#                 Optimization of Medical Inventory

#       Python using Exploratory Data Analysis and data preprocessing


# import the python libries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats 
import seaborn as sns
from sqlalchemy import create_engine

# Load the dataset from csv file

medical_inventory = pd.read_csv(r"D:\data science\optimization of Medical Inventory\Medical Inventory Optimaization Dataset\medical inventory.csv")
medical_inventory

### Credentials to connect to database 
user = 'root' ## username 
pw= '8787' ### password
db = 'optimization_medical_inventory' ### database 
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

### To_sql() - function to push the data frame into a sql table 
medical_inventory.to_sql('med_inven', con=engine, if_exists='replace', chunksize=1000, index=False)

### To_sql() - function to push the dataframe into a sql table.
sql = 'select * med_inven'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# data shape and data type

medical_inventory.shape

medical_inventory.dtypes

# type casting 

medical_inventory["Patient_ID"] = medical_inventory["Patient_ID"].astype('str')
medical_inventory["Final_Sales"] = medical_inventory["Final_Sales"].astype('float32')
medical_inventory["Final_Cost"] = medical_inventory["Final_Cost"].astype('float32')

medical_inventory.dtypes

# Handling Duplicates

duplicate = medical_inventory.duplicated()
sum(duplicate)

# Remove duplicate

medical_inventory = medical_inventory.drop_duplicates()
duplicate = medical_inventory.duplicated()
sum(duplicate)

# Handling missing values

medical_inventory.replace('', pd.NA, inplace = True)
medical_inventory.isnull().sum()

# imputation(mode)

# Check the columns in your DataFrame
print(medical_inventory.columns)

# Update group_cols based on the actual columns in your DataFrame
group_cols = ['Typeofsales', 'Patient_ID', 'Specialisation', 'Dept', 'Dateofbill']

# Impute missing values in specified columns based on the mode of the group
for col in ['Formulation', 'DrugName', 'SubCat', 'SubCat1']:
    # Extract the result of groupby operation without assigning it directly
    groupby_result = medical_inventory.groupby(group_cols)[col].apply(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
    )
    
    # Convert the result to 'object' dtype to handle potential dtype mismatch
    groupby_result = groupby_result.astype('object')
    
    # Reset the index before assigning back to the original DataFrame
    groupby_result = groupby_result.reset_index(drop=True)
    
    # Assign the result back to the original DataFrame
    medical_inventory[col] = groupby_result

# Now, medical_inventory should have missing values imputed in the specified columns

medical_inventory.isnull().sum()

# still there are some missing values that need to be dropped.

medical_inventory.dropna(inplace = True)
medical_inventory = medical_inventory.reset_index(drop = True) 
medical_inventory.isnull().sum()   
    
# Data Manipulation:
    
date_column = 'Dateofbill'
medical_inventory[date_column] = pd.to_datetime(medical_inventory[date_column], errors='coerce', infer_datetime_format=True)

# Sort dataset by date column in ascending order
medical_inventory = medical_inventory.sort_values(by=date_column, ascending=True)

# Specify Final Cost column to round
column_name = 'Final_Cost'

# Specify number of decimal places to round to 0
decimal_places = 0

# Round the values in the column to 0
medical_inventory[column_name] = medical_inventory[column_name].apply(
    lambda x: round(x, decimal_places))

# Specify Final Sales column to round
column_name1 = 'Final_Sales'

# Specify number of decimal places to round to 0
decimal_places1 = 0

# Round values in the column to 0
medical_inventory[column_name1] = medical_inventory[column_name1].apply(
    lambda x: round(x, decimal_places1))

    
medical_inventory.drop(columns=["ReturnQuantity"], axis=1, inplace=True)

medical_inventory.head(10)    
    
# Describe data

medical_inventory.describe()

## First moment business decision or Measure of central tendacy(mean, median, mode)

numeric_columns = medical_inventory.select_dtypes(include=['int64', 'float32', 'float64']).columns

mean_values = medical_inventory[numeric_columns].mean()

median_values = medical_inventory[numeric_columns].median()

mode_values = medical_inventory[numeric_columns].mode().iloc[0]

print(mean_values)
print(median_values)
print(mode_values)

# Second moment of business decision:
    
# Measure of Dispersion (variance, and standard deviation):
    
variance_values = medical_inventory[numeric_columns].var()

std_deviation_values = medical_inventory[numeric_columns].std()

print(variance_values)
print(std_deviation_values)

# Third moment of business decision(skewness)

skewness_values = medical_inventory[numeric_columns].skew()

print(skewness_values)

# Fourth moment of business decision

kurtosis_values = medical_inventory[numeric_columns].kurt()

print(kurtosis_values)

## EDA - Exploratory Data Analysis:
    
max_quantity = medical_inventory['Quantity'].max()

plt.hist(medical_inventory.Quantity, color = 'red', bins = 20, alpha = 1)
plt.xlim(0, 160)

medical_inventory.Final_Cost.max()

plt.hist(medical_inventory.Final_Cost, color = 'red', bins = 500, alpha = 1)
plt.xlim(0,3500)

medical_inventory.Final_Sales.max()

plt.hist(medical_inventory.Final_Sales, color = 'red', bins = 500, alpha = 1)
plt.xlim(0,4000)

# Positively skewed shows means greater than median

medical_inventory.RtnMRP.max()

plt.hist(medical_inventory.RtnMRP, color = 'red', bins = 100, alpha = 1)
plt.xlim(0,1000)

# Convert date formate to month

medical_inventory['Dateofbill'] = pd.to_datetime(medical_inventory['Dateofbill'])
medical_inventory['Dateofbill'] = medical_inventory['Dateofbill'].dt.strftime('%b')
medical_inventory.head()

# Pivot the DataFrame based on SubCat of drugs
stats.probplot(medical_inventory['Quantity'], dist="norm", plot=plt)
plt.show()

data_pivoted = medical_inventory.pivot_table(index="SubCat", columns="Dateofbill", values="Quantity")
# Result
data_pivoted.head()

# Data distribution:

# Distribution of data

stats.probplot(medical_inventory['Quantity'], dist="norm", plot=plt)
plt.show()

## Data Transformation: Log Transformation

# Transform the data to a normal distribution

stats.probplot(np.log(medical_inventory['Quantity']), dist="norm", plot=plt)
plt.show()

# Bar plot (quantity of durgs sold by month)

sns.barplot(data = medical_inventory, x = 'Dateofbill', y = 'Quantity')
plt.title('Quantity of drugs sold by Month')
plt.show()

# Trend in quantity:

Month = medical_inventory.groupby('Dateofbill')['Quantity'].sum()
plt.plot(Month.index, Month.values, color = 'blue')
plt.title('Quantity Trend')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.show()

# auto EDA 

# D -Tale

import dtale as dt

dt.show(medical_inventory)

d = dt.show(medical_inventory)
d.open_browser()

df_grouped = medical_inventory[['Dateofbill','Quantity']]

# Group by Quantity and Month
df_grouped = df_grouped.groupby('Dateofbill').sum()

# Result
df_grouped.head(10)
df_grouped = df_grouped.reset_index()
df_grouped

# Create dictionary to map month names into numerical values
dict_month = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Create a new column with the numerical values of the months
df_grouped['Monthindex'] = df_grouped['Dateofbill'].map(dict_month)

df_grouped = df_grouped.sort_values(by='Monthindex')

# Drop Monthindex column
df_grouped = df_grouped.drop(columns=['Monthindex'])
df_grouped = df_grouped.reset_index(drop=True)
df_grouped

# One - hot encoding 
data1 = pd.get_dummies(df_grouped.Dateofbill)
data1.columns

data = pd.concat([df_grouped , data1] , axis = 1)
data

data['log_Quantity'] = np.log(data['Quantity'])
data

data["t"] = np.arange(1,13)

data["t_square"] = data["t"] * data["t"]
data

# sweetviz auto eda:
    
import sweetviz as sv

# Create and generate a Sweetviz report
report = sv.analyze(medical_inventory)

# Save the report to an HTML file
report.show_html("auto_eda_sweetviz_report.html")

# Assuming 'medical_inventory' is your DataFrame
medical_inventory.to_csv('med inventory.csv', index = False)

# Model building 

import pickle
import csv
import seaborn as sns
import scipy.stats as stats
import pylab
from sklearn.metrics import mean_squared_error
import statsmodels.graphics.tsaplots as tsa_plots
from math import sqrt
import matplotlib.pyplot as plt

# Linear 

# Data partition

Train = data
Test = data

import statsmodels.formula.api as smf

linear = smf.ols('Quantity ~ t', data = Train).fit()
pickle.dump(linear, open('linear_model.pkl', 'wb'))

pred_linear =  pd.Series(linear.predict(pd.DataFrame(Test['t'])))
mape_linear = np.mean(np.abs((Test['Quantity'] - (pred_linear)) / Test['Quantity'])) * 100
mape_linear

## Exponential
Exponential = smf.ols('log_Quantity ~ t', data = Train).fit()
pred_Exponential = pd.Series(Exponential.predict(pd.DataFrame(Test['t'])))

mape_Exponential = np.mean(np.abs((Test['Quantity'] - np.exp(pred_Exponential)) / Test['Quantity'])) * 100
mape_Exponential

# Quadratic

Quadratic = smf.ols('Quantity ~ t + t_square', data = Train).fit()
pred_Quadratic = pd.Series(Quadratic.predict(Test[["t", "t_square"]]))

mape_Quadratic = np.mean(np.abs((Test['Quantity'] - (pred_Quadratic)) / Test['Quantity'])) * 100
mape_Quadratic

# Additive Seasonality

addSeasonality = smf.ols('Quantity ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec', data=Train).fit()
pred_addSeasonality = pd.Series(addSeasonality.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))

mape_addSeasonality = np.mean(np.abs((Test['Quantity'] - (pred_addSeasonality)) / Test['Quantity'])) * 100
mape_addSeasonality

# Multiplicative Seasonality

MulSeasonality = smf.ols('log_Quantity ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data = Train).fit()
pred_MultSeasonality = pd.Series(MulSeasonality.predict(Test))
mape_MultSeasonality = np.mean(np.abs((Test['Quantity'] - np.exp(pred_MultSeasonality)) / Test['Quantity'])) * 100
mape_MultSeasonality

# Additive Seasonality Quadratic Trend

add_seaQuadratic = smf.ols('Quantity ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_seaQuadratic = pd.Series(add_seaQuadratic.predict(Test[['Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov' , 'Oct' , 'Sep' ,'t','t_square']]))
mape_add_seaQuadratic = np.mean(np.abs((Test['Quantity'] - (pred_add_seaQuadratic)) / Test['Quantity'])) * 100
mape_add_seaQuadratic

# Multiplicative Seasonality linear Trend

Mul_SeasonalityLinear = smf.ols('log_Quantity ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data = Train).fit()
pred_MultSeasonalityLinear = pd.Series(Mul_SeasonalityLinear.predict(Test))

mape_MultSeasonalityLinear = np.mean(np.abs((Test['Quantity'] - np.exp(pred_MultSeasonalityLinear)) / Test['Quantity'])) * 100
mape_MultSeasonalityLinear

data1 = {"MODEL":pd.Series(["mape_linear","mape_Exponential","mape_Quadratic","mape_addSeasonality","mape_MultSeasonality","mape_add_seaQuadratic","mape_MultSeasonalityLinear"]),"MAPE_Values":pd.Series([mape_linear,mape_Exponential,mape_Quadratic,mape_addSeasonality,mape_MultSeasonality,mape_add_seaQuadratic,mape_MultSeasonalityLinear])}
table_mape = pd.DataFrame(data1)
table_mape

model_full = smf.ols('Quantity ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()

predict_data = data
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Quantity"] = pd.Series(pred_new)

model_full.save("model.pickle")

# Load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("model.pickle")

# Autoregression Model

# Calculating Residuals from best model applied on full data
# AV - FV
full_res = data.Quantity - model_full.predict(data)

import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA

tsa_plots.plot_acf(full_res, lags = 11)
tsa_plots.plot_pacf(full_res, lags = 5 )

# AR model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags=[1])

# model_ar = AutoReg(Train_res, lags=12)
model_fit = model_ar.fit()

print('Coefficients: %s' % model_fit.params)

pred_res = model_fit.predict(start=len(full_res), end=len(full_res)+len(predict_data)-1, dynamic=False)
pred_res.reset_index(drop=True, inplace=True)

# The Final Predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
final_pred

# ARIMA Model

train = df_grouped
test= df_grouped
train

tsa_plots.plot_acf(full_res, lags = 11)
tsa_plots.plot_pacf(full_res, lags = 5 )

model1 = ARIMA(train.Quantity, order = (5,1,2))
res1 = model1.fit()
res1.summary()

start_index = len(train)
start_index
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

forecast_test = pd.DataFrame(forecast_test)
forecast_test

from math import sqrt
from sklearn.metrics import mean_squared_error

rmse_test = sqrt(mean_squared_error(test.Quantity, forecast_test))

print('test RMSE: %.3f' % rmse_test)

plt.plot(test.Quantity)
plt.plot(forecast_test, color = 'red')
plt.show()

import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA

tsa_plots.plot_acf(full_res, lags = 11)
tsa_plots.plot_pacf(full_res, lags = 5 )

import pmdarima as pm

# Assuming you have your training data defined as 'train' with the 'Quantity' column
ar_model = pm.auto_arima(train.Quantity, start_p=0, start_q=0,
                          max_p=16, max_q=16,
                          m=1,  # frequency of series
                          d=None,  # let the model determine 'd'
                          seasonal=False,  # No Seasonality
                          start_P=0, trace=True,
                          error_action='warn', stepwise=True)

# Print ARIMA model summary
ar_model.summary()

# Predictions on test set
predictions =ar_model.predict(n_periods=len(test))

# Calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(test.Quantity, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot predictions against actual values
plt.plot(test.Quantity)
plt.plot(predictions, color='red')
plt.show()

# Auto ARIMA

model = ARIMA(train.Quantity, order = (1,0,1))
res = model.fit()
res.summary()

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res.predict(start = start_index, end = end_index)


forecast_best

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(test.Quantity, forecast_best))
print('Test RMSE: %.3f' % rmse_best)

# Plot forecasts against actual outcomes
plt.plot(test.Quantity, label='Actual')
plt.plot(forecast_best, color='red', label='Forecast')
plt.legend()
plt.show()

# Holt - winters method

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Fit an Exponential Smoothing model to the data
model = ExponentialSmoothing(df_grouped['Quantity'], seasonal_periods=4, trend='add', seasonal='mul')
model_fit = model.fit()

# Forecast for the next 12 periods
forecast = model_fit.forecast(steps=12)
forecast

# Prepare train and test datasets
train =df_grouped.Quantity
test =df_grouped.Quantity

mape= np.mean(np.abs((test - forecast) / test)) * 100
print("MAPE:",mape)

plt.plot(df_grouped['Quantity'], label='Actual')
plt.plot(forecast.index, forecast , label='Forecast')
plt.legend()
plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Fit an Exponential Smoothing model to the data
model = ExponentialSmoothing(df_grouped['Quantity'], seasonal_periods=4, trend='mul', seasonal='add')
model_fit = model.fit()

# Forecast for next 12 periods
forecast = model_fit.forecast(steps=12)
forecast

# Prepare train and test datasets
train = df_grouped.Quantity
test = df_grouped.Quantity

# Calculate root mean squared error (RMSE) of the forecast
mape= np.mean(np.abs((test - forecast) / test)) * 100
print("MAPE:",mape)

rmse = np.sqrt(mean_squared_error(test, forecast))
print("RMSE:", rmse)

plt.plot(df_grouped['Quantity'], label='Actual')
plt.plot(forecast.index, forecast , label='Forecast')
plt.legend()
plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Fit an Exponential Smoothing model to the data
model = ExponentialSmoothing(df_grouped['Quantity'], seasonal_periods=4, trend='add', seasonal='add')
model_fit = model.fit()

# Forecast for the next 12 periods
forecast = model_fit.forecast(steps=12)
forecast

# Prepare train and test datasets
train = df_grouped.Quantity
test = df_grouped.Quantity

# Calculate root mean squared error (RMSE) of the forecast
mape= np.mean(np.abs((test - forecast) / test)) * 100
print("MAPE:",mape)
rmse = np.sqrt(mean_squared_error(test, forecast))
print("RMSE:", rmse)

plt.plot(df_grouped['Quantity'], label='Actual')
plt.plot(forecast.index, forecast , label='Forecast')
plt.legend()
plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt-Winters method on training data
hw_model = ExponentialSmoothing(Train['Quantity'], seasonal_periods=6, trend='add', seasonal='add').fit()

# Predictions on test data using the fitted model
pred_hw = hw_model.forecast(len(Test))

# Calculate MAPE between predicted and actual values
mape_hw = np.mean(np.abs((Test['Quantity'] - pred_hw) / Test['Quantity'])) * 100
mape_hw

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Fit an Exponential Smoothing model to the data
model = ExponentialSmoothing(df_grouped['Quantity'], seasonal_periods=4, trend='mul', seasonal='mul')
model_fit = model.fit()

# Forecast for the next 12 periods
forecast = model_fit.forecast(steps=12)
forecast

# Prepare train and test datasets
train =df_grouped.Quantity
test =df_grouped.Quantity

# Calculate root mean squared error (RMSE) of the forecast
rmse = np.sqrt(mean_squared_error(test, forecast))
print("RMSE:", rmse)

plt.plot(df_grouped['Quantity'], label='Actual')
plt.plot(forecast.index, forecast , label='Forecast')
plt.legend()
plt.show()

# Exponential Smoothing

# Date column as index
# df_grouped.set_index('Dateofbill', inplace=True)
# Computed rolling mean of the sales data using SES

alpha =1   # smoothing parameter
df_grouped['SES'] = df_grouped['Quantity'].ewm(alpha=alpha, adjust=False).mean()

plt.plot(df_grouped['Quantity'], label='Actual')
plt.plot(forecast.index, forecast , label='Forecast')
plt.legend()
plt.show()

# Plot  of original sales data and the SES forecast
plt.plot(df_grouped['Quantity'], label='Actual Sales')
plt.plot(df_grouped['SES'], label='SES Forecast')
plt.legend()
plt.show()

# Random forest and Linear Regression Model

df = data[['Dateofbill', 'Quantity']]

df.index.freq = 'MS'

df.head(10)

df.set_index('Dateofbill')

df.plot(figsize=(10,6))

df['Quantity_LastMonth']=df['Quantity'].shift(+1)
df['Quantity_2Monthsback']=df['Quantity'].shift(+2)
df['Quantity_3Monthsback']=df['Quantity'].shift(+3)
df

df = df.dropna()
df

from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100,max_features=3, random_state=1)

import numpy as np
x1,x2,x3,y=df['Quantity_LastMonth'],df['Quantity_2Monthsback'],df['Quantity_3Monthsback'],df['Quantity']
x1,x2,x3,y=np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y=x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x=np.concatenate((x1,x2,x3),axis=1)
print(final_x)

X_train,X_test,y_train,y_test=final_x[:],final_x[-10:],y[:],y[-10:]

model.fit(X_train,y_train)
lin_model.fit(X_train,y_train)

# Random Forest Regression

pred = model.predict(X_test)
plt.rcParams["figure.figsize"] = (10, 5)
plt.plot(pred, label='Random_Forest_Predictions')
plt.plot(y_test, label='Actual Quantity')
plt.legend(loc="upper left")
plt.show()

lin_pred=lin_model.predict(X_test)
plt.rcParams["figure.figsize"] = (10,5)
plt.plot(lin_pred, label='Linear_Regression_Predictions')
plt.plot(y_test,label='Actual Quantity')
plt.legend(loc="upper left")
plt.show()  

rmse_rf=sqrt(mean_squared_error(pred,y_test))
rmse_lr=sqrt(mean_squared_error(lin_pred,y_test))

print('Mean Squared Error for Random Forest Model is:',rmse_rf)
print('Mean Squared Error for Linear Regression Model is:',rmse_lr)

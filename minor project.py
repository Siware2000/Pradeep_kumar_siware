#!/usr/bin/env python
# coding: utf-8

# In[6]:



#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
df = pd.read_csv('minor.csv.csv')

#exploring the dataset
df.head()
df.info()
df.describe()

#visualizing the dataset
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Stock Price Prediction')
plt.show()

#splitting the dataset into training and testing sets
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error:', rmse)
print('R2 Score:', r2)


# In[ ]:





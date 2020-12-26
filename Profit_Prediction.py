# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('online.csv')
print(df)
print(df.shape)

# checking the value is null or not
print(df.isnull().sum())

# as profit our output column so we drop it from dataset
x = df.drop(['Profit'], axis=1)
print(x)

# then store the profit column into y variable
y = df['Profit']
print(y)

# Using one hot encoding
# Convert the columns into numerical columns
city = pd.get_dummies(x['Area'], drop_first=True)
print(city)

# Drop the area column
x = x.drop('Area', axis=1)
print(x)

# Concatenation
x = pd.concat([x, city], axis=1)
print(x)

# Import Library
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Fitting multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(y_test)

# Predicting the test set result
pred = regressor.predict(x_test)
print(pred)

# Accuracy
print(regressor.score(x_test, y_test))

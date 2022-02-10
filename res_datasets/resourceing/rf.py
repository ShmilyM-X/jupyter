from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas
import os
import numpy as np

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def calcrae(p, a):
    suma = 0.0
    for i in range(len(p)):
        suma += a[i]
    ave = suma/len(p)
    f1 = f2 = 0.0
    for i in range(len(p)):
        # f1+=abs(p[i]-a[i])
        # f2+=abs(ave-a[i])
        f1 += (p[i]-a[i])*(p[i]-a[i])
        f2 += (ave-a[i])*(ave-a[i])
    return (f1**0.5)/(f2**0.5)


dataframe = pandas.read_csv("./lable.csv", header=0)
dataset = dataframe.values
dim = 61
datas = dataset[0:, 1:dim+1].astype('float')
labels = dataset[0:, dim+1].astype('float')
names = dataset[0:, 0]
print(datas)
print(labels)

x_train = datas
y_train = labels
# scaler = preprocessing.MinMaxScaler()
scaler =preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)

# x_val = x_train[0:255]
# partial_x_train = x_train[255:]
# y_val = y_train[0:255]
# partial_y_train = y_train[255:]
x_val, partial_x_train, y_val, partial_y_train = train_test_split(
    x_train, y_train, test_size=0.2, random_state=1)

reg = RandomForestRegressor(random_state=42)
reg.fit(x_val, y_val)

result = reg.predict(partial_x_train)
for i in range(len(result)):
    print("%s %.2f %d" % (names[i], result[i], partial_y_train[i]))
print(r2_score(partial_y_train, result))
print(mean_squared_error(partial_y_train, result))
print(mean_absolute_error(partial_y_train, result))
print(calcrae(partial_y_train, result))

# result=reg.predict(x_train)
# # for i in range(len(result)):
# #     print ("%s %.2f %d" % (names[i],result[i],y_train[i]))
# print (r2_score(y_train,result))
# print (mean_squared_error(y_train,result))
# print (mean_absolute_error(y_train,result))
# print (calcrae(y_train,result))


linreg = LinearRegression()
model = linreg.fit(x_val, y_val)
result = linreg.predict(partial_x_train)

print(linreg.coef_)

for i in range(len(result)):
    print("%s %.2f %d" % (names[i], result[i], partial_y_train[i]))
print(r2_score(partial_y_train, result))
print(mean_squared_error(partial_y_train, result))
print(mean_absolute_error(partial_y_train, result))
print(calcrae(partial_y_train, result))

# result=linreg.predict(x_train)
# # for i in range(len(result)):
# #     print ("%s %.2f %d" % (names[i],result[i],y_train[i]))
# print (r2_score(y_train,result))
# print (mean_squared_error(y_train,result))
# print (mean_absolute_error(y_train,result))
# print (calcrae(y_train,result))

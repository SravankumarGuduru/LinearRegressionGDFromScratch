import pandas as pd
import numpy as np
import math 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://drive.google.com/uc?id=1ssWzzmCpjo9sqnxUq9VyyB61s8rojpg3")

df = df.drop(df[df["horsepower"] == "?"].index)
df = df.astype(dtype={"horsepower":"int"})
# print(df.head(2))
# print(len(df))

del df["car name"]
# df.columns


def normalize(data):
    for col in data.columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()
        data[col] = np.exp(-(data[col] - data[col].mean()) ** 2 / (2 * (data[col].std()) ** 2))
    return data

def split_data(df):
    X = df[['cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'model year', 'origin']]
    Y = df["mpg"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = split_data(df)

X_train = normalize(X_train)
X_test = normalize(X_test)


X0_train = np.ones([len(X_train),1])
X0_test = np.ones([len(X_test),1])
X_train = np.append(X0_train, X_train, axis=1)
X_test = np.append(X0_test, X_test, axis=1)


def test(X_test, Y_test, model: LinearRegression):
    Y_pred = model.predict(X_test)
    evaluate_model(Y_pred, Y_test)


def train(X_train, Y_train):
    reg = LinearRegression()
#     weight = np.zeros(X_train.shape[1], dtype=float)
    # Data Fitting
    reg = reg.fit(X_train, Y_train)
    # Y Prediction
    Y_pred = reg.predict(X_test)

    evaluate_model(Y_pred, Y_test)
    return reg


def evaluate_model(pred, true):
    errors = np.subtract(pred, true)
    mse = 1 / (2 * true.shape[0]) * errors.T.dot(errors)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    print('MSE is {}'.format(mse))
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")

model = train(X_train, Y_train)
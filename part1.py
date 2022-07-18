
import numpy as np
import pandas as pd
import logging
import math 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


logging.basicConfig(filename="part1.log", level=logging.INFO, format='%(message)s')

class LinearRegression:
    def __init__(self, df):
        self.df = df

    def normalize(self,data):
        for col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
            data[col] = np.exp(-(data[col] - data[col].mean()) ** 2 / (2 * (data[col].std()) ** 2))
        return data

    def split_data(self,df):
        X = df[['cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'model year', 'origin']]
        Y = df["mpg"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
        return X_train, X_test, Y_train, Y_test


    def train_error(self,X_train,Y_train,weight,m):
            predictions = np.dot(X_train,weight)
            errors = predictions - Y_train
            mse = 1 / (2 * m) * errors.T.dot(errors)
            return mse

    def gradient_descent(self,X_train, Y_train,X_test,Y_test, weight, learning_rate=0.01, iterations=1000):
        m = len (Y_train)
        mse = np.zeros(iterations, dtype=float)
        test_mse = np.zeros(iterations, dtype=float)

        for i in range (iterations):
            predictions= np.dot(X_train, weight)
            errors = np.subtract(predictions, Y_train)
            weight = weight - (1/m) * learning_rate * (X_train.T.dot(errors))
            mse[i] = self.train_error(X_train,Y_train,weight,m)
            test_mse[i] = self.test_final(weight,X_test,Y_test)
        return weight, mse,test_mse

    def test_final(self,weight, X_test, Y_test, return_only_mse = True):
        predictions = X_test.dot(weight)
        errors = predictions - Y_test

        mse = 1 / (2 * X_test.shape[0]) * errors.T.dot(errors)
        if return_only_mse == True:
            return mse
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, predictions)
        return mse,rmse,r2

    def data_processing(self,df):
        X_train, X_test, Y_train, Y_test = self.split_data(df)

        X_train = self.normalize(X_train)
        X_test = self.normalize(X_test)

        X0_train = np.ones([len(X_train),1])
        X0_test = np.ones([len(X_test),1])
        X_train = np.append(X0_train, X_train, axis=1)
        X_test = np.append(X0_test, X_test, axis=1)

        return X_train, X_test, Y_train, Y_test

    def fit(self,learning_rate = 0.001, iterations = 5000):
        X_train, X_test, Y_train, Y_test = self.data_processing(self.df)

        # weight = np.zeros((8,1))
        # weight_1 = np.array([[ 0.0],[0.0],[0.],[0.0],[0.0],[0.0],[ 0.0],[ 0.0]])

        weight = np.zeros(X_train.shape[1], dtype=float)
        weight, mse,test_mse = self.gradient_descent(X_train, Y_train,X_test,Y_test, weight, learning_rate, iterations)

        final_mse,rmse,r2 = self.test_final(weight,X_test,Y_test,False)
        # final_mse = self.test_final(weight,X_test,Y_test)clear
        
        logging.info("=================================")
        logging.info(f"learning_rate = {learning_rate}, iterations = {iterations} ")
        logging.info(f"The model performance for testing data is {final_mse}")
        logging.info('RMSE is {}'.format(rmse))
        logging.info('R2 score is {}'.format(r2))
        logging.info("final weights after traing are: {}".format(' '.join(map(str, weight))))
        logging.info("")
        logging.info("")

        return weight,mse,test_mse,final_mse

def download_data():
    # df = pd.read_csv("./auto-mpg.csv")
    df = pd.read_csv("https://drive.google.com/uc?id=1ssWzzmCpjo9sqnxUq9VyyB61s8rojpg3")


    df = df.drop(df[df["horsepower"] == "?"].index)
    df = df.astype(dtype={"horsepower":"int"})

    del df["car name"]
    return df
    
def main():
    df = download_data()
    linearRegression = LinearRegression(df)

    linearRegression.fit(learning_rate=0.001, iterations=1000)
    linearRegression.fit(learning_rate=0.001, iterations=10000)
    linearRegression.fit(learning_rate=0.01, iterations=10000)
    linearRegression.fit(learning_rate=0.05, iterations=10000)
    linearRegression.fit(learning_rate=0.02, iterations=10000)
    linearRegression.fit(learning_rate=0.09, iterations=1000)
    linearRegression.fit(learning_rate=0.09, iterations=10000)
    linearRegression.fit(learning_rate=0.09, iterations=750)


if __name__ == '__main__':
    main()
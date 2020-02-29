import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import math


# style.use('fivethirtyeight')

class LinearRegression():
    def __init__(self, raw_X, y):
        self.raw_X = raw_X
        self.y = y
        if len(raw_X) != len(y):
            warnings.warn("Size of X features and Y outputs are not equal!")
            raise ValueError

        self.train_data, self.test_data = self.train_test_split(raw_X, y)
        self.train_X = self.train_data[:, :-1]
        self.test_X = self.test_data[:, :-1]
        self.train_y = self.train_data[:, -1].reshape((-1, 1))
        self.test_y = self.test_data[:, -1].reshape((-1, 1))
        self.raw_X_train = self.train_X
        self.raw_X_test = self.test_X

    def train_test_split(self, raw_X, y):

        data = np.c_[raw_X, y]
        np.random.shuffle(data)
        data = np.c_[np.ones((len(raw_X),1)),data]


        test_size = 0.2
        train_set = list()
        test_set = list()
        train_data = data[:-int(len(data) * test_size)]
        test_data = data[-int(len(data) * test_size):]

        return train_data, test_data

    def normal_equation(self):

        # Regularization for N.E
        Lambda = 0.03
        reg_matrix = np.identity(len(self.train_X[0]))
        reg_matrix[0][0] = 0

        self.theta_best = np.linalg.inv(self.train_X.T.dot(self.train_X) + Lambda * reg_matrix).dot(self.train_X.T).dot(self.train_y)

    def batch_gradient_descent(self):
        alpha = 0.8
        m = len(self.train_X)
        theta = np.random.randn(len(self.train_X[0]), 1)

        if len(self.train_X[0]) <= 2:
            self.train_X, self.test_X = self.MinMaxScaler(self.train_X, self.test_X)
        elif len(self.train_X[0]) > 2:
            self.train_X, self.test_X = self.MultipleMinMaxScaler(self.train_X, self.test_X)

        # For regularization in batch g.d
        Lambda = 0.01

        # Loop for comparison between gradients
        gradient_temp = 2 * np.dot(self.train_X.transpose(), np.dot(self.train_X, theta) - self.train_y)
        x = 1
        while True:

            hypothesis = np.dot(self.train_X, theta)

            gradient = np.dot(self.train_X.transpose(), hypothesis - self.train_y)

            if (sum(abs(gradient_temp)) * 100 / abs(sum(gradient))) - 100 <= 0.00001:
                break

            gradient_temp = gradient

            theta = theta - (alpha * gradient) / m - Lambda / m * theta
            x += 1
        print(f"No of iterations={x}")

        """
        #Iteration form

        for i in range(350):
            hypothesis = np.dot(self.train_X,theta)

            gradients = np.dot(self.train_X.transpose(),hypothesis - self.train_y)
            theta = theta - (alpha * gradients)/m - Lambda/m * theta

        """
        self.theta_Scaled = theta

        for i in range(1,len(theta)):
            theta[i] = theta[i] / (self.X_Max[i-1] - self.X_Min[i-1])
        for i in range(1,len(theta) -1):
            theta[0] = theta[0] - (theta[i] * self.X_Min[i-1])

        self.theta_best = theta

    def r2_Accuracy(self):

        prediction = self.raw_X_test.dot(self.theta_best)

        RSS = [((self.test_y[i] - prediction[i]) ** 2) for i in range(len(self.test_y))]
        TSS = [((self.test_y[i] - self.test_y.mean()) ** 2) for i in range(len(self.test_y))]
        if (sum(TSS)) == 0:
            r2 = 1
        else:
            r2 = (1 - (sum(RSS) / sum(TSS)))

        print(f"R Squared={r2}")
        RMSE = math.sqrt((sum(RSS)) / len(self.test_y)) / self.test_y.mean()
        print(f"Rmse={RMSE}")
        print(f"Accuracy(RMSE) = {1 - RMSE}")


    def MultipleMinMaxScaler(self, train_X, test_X):
        X_Full = np.r_[train_X[:,1:],test_X[:,1:]]
        self.X_Min = [min(X_Full[:,column]) for column in range(len(X_Full[0]))]
        self.X_Max = [max(X_Full[:,column]) for column in range(len(X_Full[0]))]

        for i in range(len(X_Full[0])):
            for j in range(len(X_Full)):
                X_Full[j][i] = (X_Full[j][i] - self.X_Min[i]) / (self.X_Max[i] - self.X_Min[i])

        X_w_ones = np.c_[np.ones((len(X_Full),1)),X_Full]
        X_train = X_w_ones[:-len(test_X)]
        X_test = X_w_ones[-len(test_X):]


        return X_train,X_test

    def show_parameters(self):
        print(f"y_intercept={self.theta_best[0]}")
        for i in range(1,len(self.theta_best)):
            print(f"x{i}={self.theta_best[i]}")

    def predict(self, x):
        X = np.c_[np.ones(len(x), 1), x]
        prediction = self.theta_best.T.dot(X)
        return prediction

    def plot(self):
        #plt.scatter(range(len(self.test_X)),self.test_X.dot(self.theta_Scaled),color='r')
        plt.scatter(range(len(self.raw_X_test)),self.raw_X_test.dot(self.theta_best),color='r')
        plt.scatter(range(len(self.raw_X_test)),self.test_y,color='b')


X = 2 * np.random.rand(250, 2)
y = (4 + np.sum(3 * X) + np.random.randn(250, 1))

regressor = LinearRegression(X, y)
regressor.batch_gradient_descent()
#regressor.normal_equation()
regressor.r2_Accuracy()
regressor.show_parameters()
regressor.plot()

















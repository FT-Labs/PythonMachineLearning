import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import math

#style.use('fivethirtyeight')

class LinearRegression():
    def __init__(self,raw_X,y):
        self.raw_X = raw_X
        self.y = y
        if len(raw_X) != len(y):
            warnings.warn("Size of X features and Y outputs are not equal!")
            raise ValueError

        self.train_data, self.test_data = self.train_test_split(raw_X,y)
        self.train_X = self.train_data[:,:-1]
        self.test_X = self.test_data[:,:-1]
        self.train_y = self.train_data[:,-1].reshape((-1,1))
        self.test_y = self.test_data[:,-1].reshape((-1,1))
        self.raw_X_train = self.train_X
        self.raw_X_test = self.test_X

    def train_test_split(self,raw_X,y):

        self.X_min = min(raw_X)
        self.X_max = max(raw_X)

        X = np.c_[np.ones((len(raw_X),1)),raw_X]
        data = np.c_[X,y]
        np.random.shuffle(data)

        test_size = 0.2
        train_set = list()
        test_set = list()
        train_data = data[:-int(len(data)*test_size)]
        test_data = data[-int(len(data)*test_size):]

        return train_data,test_data

    def normal_equation(self):

        #Regularization for N.E
        Lambda = 0.03
        reg_matrix = np.identity(len(self.train_X[0]))
        reg_matrix[0][0] = 0


        self.theta_best = np.linalg.inv(self.train_X.T.dot(self.train_X) + Lambda * reg_matrix).dot(self.train_X.T).dot(self.train_y)

    def batch_gradient_descent(self):
        alpha = 0.1
        m = len(self.train_X)
        theta = np.random.randn(2,1)

        self.train_X,self.test_X = self.MinMaxScaler(self.train_X,self.test_X)

        #For regularization in batch g.d
        Lambda = 0.01

        #Loop for comparison between gradients
        gradient_temp = 2 * np.dot(self.train_X.transpose(), np.dot(self.train_X, theta) - self.train_y)
        x = 1
        while True:

            hypothesis = np.dot(self.train_X,theta)

            gradient = np.dot(self.train_X.transpose(),hypothesis - self.train_y)

            if (abs(sum(gradient_temp)) * 100 / abs(sum(gradient))) - 100 <= 0.01:
                break

            gradient_temp = gradient

            theta = theta - (alpha * gradient)/m - Lambda/m * theta
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

        theta[1] = (theta[1]/(self.X_max-self.X_min))
        theta[0] = theta[0]- (theta[1] * self.X_min)

        self.theta_best = theta

    def r2_Accuracy(self):

        prediction = self.raw_X_test.dot(self.theta_best)

        RSS = [((self.test_y[i] - prediction[i])**2) for i in range(len(self.test_y))]
        TSS = [((self.test_y[i] - self.test_y.mean())**2) for i in range(len(self.test_y))]
        if (sum(TSS)) == 0:
            r2 = 1
        else:
            r2 = (1 - (sum(RSS)/sum(TSS)))


        print(f"R Squared={r2}")
        RMSE = math.sqrt((sum(RSS))/len(self.test_y)) / self.test_y.mean()
        print(f"Rmse={RMSE}")
        print(f"Accuracy(RMSE) = {1-RMSE}")
        print(f"y_intercept={self.theta_best[0]},m={self.theta_best[1]}")

    def MinMaxScaler(self,train_X,test_X):

        norm_train_X = [(train_X[i][1] - self.X_min) / (self.X_max - self.X_min) for i in range(len(train_X))]
        norm_test_X = [(test_X[i][1] - self.X_min) / (self.X_max - self.X_min) for i in range(len(test_X))]

        norm_train_X = np.c_[np.ones((len(norm_train_X),1)),norm_train_X]
        norm_test_X = np.c_[np.ones((len(norm_test_X),1)), norm_test_X]


        return norm_train_X,norm_test_X



    def predict(self,x):
        X = np.c_[np.ones(len(x),1),x]
        prediction = self.theta_best.T.dot(X)
        return prediction
    def plot(self):
        plt.scatter(self.raw_X_train[:,1],self.train_y,marker="o")
        #plt.scatter(self.raw_X_test[:,1],np.dot(self.raw_X_test,self.theta_best),marker="x")
        plt.scatter(self.raw_X_test[:,1],self.test_y,marker="d")
        plt.plot(self.raw_X_train[:,1],np.dot(self.raw_X_train,self.theta_best),color='r')
        plt.title('Regression from Scratch')

        plt.show()



X = 2 * np.random.rand(250,1)
y = 4 + 3 * X + np.random.randn(250,1)




regressor = LinearRegression(X,y)
regressor.batch_gradient_descent()
#regressor.normal_equation()
regressor.r2_Accuracy()
#regressor.plot()

















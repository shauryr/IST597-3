import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
IST 597: Foundations of Deep Learning
Problem 2: Polynomial Regression & 

@author - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit'  # will add a unique sub-string to output of this program
degree = 15  # p, order of model
beta = 0.1  # regularization coefficient
alpha = 1  # step size coefficient
eps = 0.00001  # controls convergence criterion
n_epoch = 10000  # number of epochs (full passes through the dataset)
folder_prefix = './out/prob2_folder/'


# begin simulation

def regress(X, theta):
    # WRITEME: write your code here to complete the routine
    ##############################################################################
    X = feature_map(X)
    b, w = theta
    y = b + np.dot(X, w.T)
    ##############################################################################
    return y


def gaussian_log_likelihood(mu, y):
    # WRITEME: write your code here to complete the routine
    return -1.0


def computeCost(X, y, theta, beta):  # loss is now Bernoulli cross-entropy/log likelihood
    # WRITEME: write your code here to complete the routine
    ##############################################################################
    m = len(X)
    _, w = theta
    j_theta = (np.sum((regress(X, theta) - y) ** 2) / (2 * m)) + (beta * np.sum(w ** 2)) / (2 * m)
    ##############################################################################
    return j_theta


def computeGrad(X, y, theta, beta):
    # WRITEME: write your code here to complete the routine (
    # NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    ##############################################################################
    _, w = theta
    m = len(X)
    dL_dfy = None  # derivative w.r.t. to model output units (fy)
    dL_db = (np.sum(regress(X, theta) - y)) / m  # derivative w.r.t. model weights w
    f_x = regress(X, theta)
    dL_dw = np.dot((f_x - y).T, feature_map(X)) / m + ((w * beta) / m)  # derivative w.r.t model bias b
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    ##############################################################################
    return nabla


path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)


# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)
##############################################################################
def feature_map(X):
    poly_X = []
    for j in xrange(len(X)):
        X_list = []
        for i in range(1, degree + 1):
            X_list.append(X[j][0] ** i)
        poly_X.append(X_list)
    return np.array(poly_X)


##############################################################################

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1, degree))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta, beta)
print("-1 L = {0}".format(L))
i = 0
halt = 0
cost = [L]
while (i < n_epoch and halt == 0):
    dL_db, dL_dw = computeGrad(X, y, theta, beta)
    b = theta[0]
    w = theta[1]
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ##############################################################################
    b = b - alpha * dL_db
    w = w - alpha * dL_dw
    theta = (b, w)
    L = computeCost(X, y, theta, beta)
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    if cost[- 1] - L < eps:
        break
    cost.append(L)
    ##############################################################################
    print(" {0} L = {1}".format(i, L))
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test,
                        axis=1)  # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))

# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)
##############################################################################
plt.plot(X_test, regress(X_feat, theta), label="Model")
##############################################################################
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
##############################################################################
plt.savefig(folder_prefix + 'scatterplot_model_prob2_alpha_' + str(alpha) + '.png')
plt.show()
plt.plot(cost, 'r')
plt.savefig(folder_prefix + 'loss_alpha_' + str(alpha) + '.png')
plt.show()
##############################################################################

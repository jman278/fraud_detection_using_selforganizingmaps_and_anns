

# Part 1 - Identify the Frauds with the Self-Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values # this is used only to compare at the end the predictions made agaainst the actual; not used while buildiing the SOM

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom # requires minisom.py file in the same folder
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) # SOM will be a 10 by 10 grid (arbitrary choice); 15 features are present in training data X; radius is 1; higher the LR, faster the convergence
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone() # initializes the window containing the map
pcolor(som.distance_map().T) # adds the inter-neuron distance for all the winning nodes of the SOM
colorbar() # add the legend of the colors added above
# adding markers (red circles and green squares) to the winning nodes to check if they got approval or not
markers = ['o', 's'] 
colors = ['r', 'g']
for i, x in enumerate(X): # loop over each customer
    w = som.winner(x) # obtain winning node for the customer
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2) # color the node depending on whether the customer got approval or not
show() # red circles correspond to customers who didn't get approval and green squares corresponds to the ones who did

# Finding the frauds 
mappings = som.win_map(X) # returns a dictionary mapping the winning nodes to the customers
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0) # the coordinates of the outlying winning nodes from the plot above are passed to get the list of fraudulent customers
frauds = sc.inverse_transform(frauds) # inverse mapping to get the original unscaled customer IDs


# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset)) # create a zero vector of length equal to the number of rows in the dataset
# replace zeroes for customers who are potential fraudelents
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds: # if customer ID is in frauds
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Build the ANN and fit it to our training

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers) # get the predicted probabilities
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1) # create a dataset with the Customer IDs along with predicted probabilities
y_pred = y_pred[y_pred[:, 1].argsort()] # sort the dataset by predicted probabilities


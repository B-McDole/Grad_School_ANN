# Grad School Data

# Part 1 Preprocessing

# Importing the libraries
import numpy as np 
# We need numpy to at some point work with arrays
import pandas as pd 
# Pandas actually lets us read our csv!
import os 
# os is imported to let us change our directory and not have
# to manually set our directory repeatedly!

path = "/Users/benjaminmcdole/Desktop/" 
# Changing the path just allows us 
# to write and read files with a bit shorter path
os.chdir(path)

# Importing the dataset
dataset = pd.read_csv('Admit_Predict.csv')
# we use pd here to essentially let python know where to look for read_csv()
dataset2 = dataset.dropna()
# Yes we could probably do some work and fix up our missing pieces
# For now we will work solely on creating the model.  As such we toss missing values
X = dataset2.iloc[:, 1:8].values
# Quick thing to note.  We cut row 8 because it is what we are trying to predict!
# Another quick thing to note, rows "1 through 8" actually means we are starting at
# GRE score and not index.  Awkward, but I clarify a bit below
# Also of note, we need iloc to return values.  Additionally, [:,:] notation is;
# [1:8, 2:12] would be ROWS 1-7, and columns 2-11.  We do not include the last boundary.
# Also a reminder to save you a headache, Python (like most OOP languages) starts
# Counting at 0 instead of 1
y = dataset2.iloc[:, 8].values
# Just the y-value, so only column 8, but that : means all rows

# We only have one categorical variable, so we don't need to encode anything.
# If we did need to do so, here is some information on how to do it (It's... not bad)
# Encoding categorical Independent Variables
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#
#ct = ColumnTransformer(
#    [('one_hot_encoder', OneHotEncoder(), [8])],    # The column numbers to be transformed (here is [8] but can be [0, 1, 3])
#    remainder='passthrough'                         # Leave the rest of the columns untouched
#)
#X = np.array(ct.fit_transform(X), dtype=np.float)
#X = X[:, 1:] # This is here to cut out one of the columns, to avoid the dummy variable trap
# The dummy variable trap is a situation that happens when we encode categories
# Really, when we add categories we run the risk of confusing the model with coefficients
# The moral of the story is, if you encode a category, cut one of the columns out

from sklearn.model_selection import train_test_split
# How convenient!  Train/test split is really nicely done here, all in one step
# 80/20 still pretty much the standard for splitting.  random.state is what keeps our
# "random" consistent and not actually that random
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# We need to scale our data.  There are different methods you can use
# We use essentially a standard distribution.  Also of note, the 'standard scaler'
# will find mean and standard deviation on the train set, and then can be used on anything
X_train = sc.fit_transform(X_train)
# fit_transform finds the mean and standard deviation, and then also transforms the data
X_test = sc.transform(X_test)
# transform uses the information for sc found above and then just transforms the data

# Fixing up the y_values so we can use accuracy
y_train = (y_train > 0.5)
# This is kind of like ifelse in R
# If y_train is greater than 0.5, True, otherwise False (1, otherwise 0)

# Making the ANN
import keras
from keras.models import Sequential 
# Need to initialize model
# Because this is a straightforward model, we are adding layers sequentially
from keras.layers import Dense 
# Needed to create layers

# Initializing the ANN
classifier = Sequential()
# We are creating a classifier, yes or no, did they get in.
# This is why we "fixed" the y_train values above.
  
# Note output = 1 since this is just returning probability
# units are the number of neurons in the hidden layer
# kernel_initializer is how we distribute the inital weights.  There are a lot of
# options, but we will go uniform, essentially all around 0.
# We also need an activation function, to take us to the hidden layer.
# Again there are some common options.  We will use 'relu'
# rectified linear unit, essentially a function that is 0 for a while, and then becomes a line
# Last thing!  input_dim is of course our input dimensions.
classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu', input_dim=7))

#How many neurons do we use?  How many hidden layers?  GOOD QUESTION
#There is no single firm answer.  Here are some quick guidelines:
#The number of hidden neurons should be between the size of the input layer,
# and the size of the output layer
#The number of hidden neurons should be 2/3 the size of the input layer, plus the,
# size of the output layer
#The number of hidden neurons should be less than twice the size of the input layer

# Adding second hidden layer
classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu')) 
#Units, number of nodes in hidden layer.
# Notice we don't need the input_dim because that's just the input layer.
# The activation function could change here if we wanted.  No need to do so though

# Adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# Output of 1, probability.  Need to change activation to sigmoid
# Last we have the output layer.  Just a single unit because we are returning a simple 0 or 1
# The sigmoid function is probably the best way to go from 0 to 1 in this case,
# though again there are other functions we could use

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# We get to compile our function!  That's great, because it means we are almost to the cool part
# ADAM is adapted from adaptive moment generation, and we use it instead of the gradient
# Gradients are also very commonly used
# We use binary_crossentropy because we are predicting a binary outcome
# The metric we use is accuracy, it serves us well here.  We would, in the course
# of selecting our model, try other metrics (though accuracy is generally pretty good)

# Fitting ANN to the Training Set
classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=200) 
# The beauty behind ANNs is how quickly and frequently weights are evaluated
# We could evaluate weights after going through all of the samples in the data
# Instead we will revisit the weights every 10 elements; Again worth noting
# we would check batch sizes during tuning
# epochs are how many cycles of the full data set we will run through.
# Typically 100-200 or so are good for an initial workup; but like batch_size this
# would be tuned later on for optimal outcomes
# Also something to note, watching the classifier train is VERY COOL
# One other thing of note:
#We probably have more layers than we need, and we probably have a small set
#for this many layers.  Our model will not always train this fast, I promise

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Ah yes, our old friend 'predict'.  Used in Python and R because it just makes sense
# We are doing our best to predict the likelihood of getting accepted.  We will
# now just smooth out the rough edges

y_pred = (y_pred > 0.5)
y_test = (y_test > 0.5)
# Changing both the values, as we did above
# Our last accuracy was around 95%
# What happens when we test this model?

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Pretty standard confusion matrix, just like in R and other areas
# Confusion matrix shows us we nailed it 91 times, for an average of about
# ~93%, fantastic!

# There are a ton of details we could add here.  In fact, I will probably go
#and add some of those details in a second pass.  Things like predicting
#a single instance (just 1 student), for example.  I was also a little more loose
#on the details than I planned to be.  There are A LOT of small things that can
#disrupt the flow of work if all handled.  I try and find the balance.













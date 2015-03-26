import numpy as np
from sklearn.cluster import KMeans

# import the data
dat = np.genfromtxt('input.csv', delimiter = ',')

# shuffle the rows
np.random.shuffle(dat)

# wish to train on approximately 90%
trainLen = np.floor(0.9 * dat.shape[0])
train = dat[0:trainLen, 1:]
test = dat[trainLen:, 1:]
labels = dat[trainLen:, 0]

# initialize the object
km = KMeans(2)
km.fit(train)
guess = km.predict(test)

# evaluate accuracy
print "Using 90% of the data to train our clusters, then testing on the \n \
remaining 10%, we report an accuracy of " + str(np.mean(guess == labels))

km = KMeans(2)
guess = km.fit_predict(dat[:,1:])

# use built-in fit_predict
print "Alternatively, using the built-in fit_predict function, our accuracy \n \
is " + str(np.mean(guess == dat[:,0]))

# conclusion
print "This leads us to believe that perhaps we should transform the data \n \
before attempting to cluster the data."
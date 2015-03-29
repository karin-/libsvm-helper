import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import the data
dat = np.genfromtxt('data.csv', delimiter = ',')

# shuffle the rows
np.random.shuffle(dat)

# wish to train on approximately 90%
trainLen = np.floor(0.9 * dat.shape[0])
train = dat[0:trainLen, 1:]
test = dat[trainLen:, 1:]
labels = dat[trainLen:, 0]

km = KMeans(2)
guess = km.fit_predict(dat[:,1:])

# use built-in fit_predict
print "Alternatively, using the built-in fit_predict function, our accuracy \n \
is " + str(np.mean(guess == dat[:,0]))

# Now for PCA
# initialize the object
pca1 = PCA(1)
pca1.fit(dat[:,1:])
print "explained variance ratio with 1 component: " + \
    str(sum(pca1.explained_variance_ratio_))

pca2 = PCA(2)
pca2.fit(dat[:,1:])
print sum(pca2.explained_variance_ratio_)
print "explained variance ratio with 2 components: " + \
    str(sum(pca2.explained_variance_ratio_))

pca3 = PCA(3)
pca3.fit(dat[:,1:])
print sum(pca3.explained_variance_ratio_)
print "explained variance ratio with 3 components: " + \
    str(sum(pca3.explained_variance_ratio_))

pca10 = PCA(10)
pca10.fit(dat[:,1:])
print sum(pca10.explained_variance_ratio_)
print "explained variance ratio with 10 components: " + \
    str(sum(pca10.explained_variance_ratio_))

# Using 3 components heuristically as that gets us most of the variance

pcaData = pca3.fit_transform(dat[:,1:])

km = KMeans(2)
guess = km.fit_predict(pcaData)

# use built-in fit_predict
print "Using the built-in fit_predict function, our accuracy \n \
is " + str(np.mean(guess == dat[:,0]))

# lets plot things!
import matplotlib.pyplot as mpl

colors = np.zeros((dat.shape[0], 3))
for row in range(colors.shape[0]):
    if dat[row,0]:
        colors[row,:] = [1, 0, 0]
    else:
        colors[row,:] = [0, 0, 1]


mpl.scatter(pcaData[:,0], pcaData[:,1], c = colors)
mpl.show()

mpl.scatter(pcaData[:,0], pcaData[:,2], c = colors)
mpl.show()

mpl.scatter(pcaData[:,1], pcaData[:,2], c = colors)
mpl.show()
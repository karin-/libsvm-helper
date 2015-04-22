import numpy as np
import nmf.py
import matplotlib.pyplot as mpl

transformedData = nmf.fit_transform(tfidf)

# colorize that data!
colors = np.zeros((transformedData.shape[0], 3))
for row in range(colors.shape[0]):
    if row < len(techList):
        # tech is red
        colors[row,:] = [1, 0, 0]
    else:
        # health is blue
        colors[row,:] = [0, 0, 1]


# plot that data!
mpl.scatter(transformedData[:,0], transformedData[:,1], c = colors)
mpl.show()


# cluster that data!
from sklearn.cluster import KMeans
km = KMeans(2)
correct = np.repeat([0,1], [len(techList), len(healthList)])
guess = km.fit_predict(transformedData)

print "Using the built-in fit_predict function, our accuracy is " + str(np.mean(guess == correct))



# colorize that data!
colors2 = np.zeros((transformedData.shape[0], 3))
for row in range(colors2.shape[0]):
    if guess[row]:
        # tech is red
        colors2[row,:] = [1, 0, 0]
    else:
        # health is blue
        colors2[row,:] = [0, 0, 1]

mpl.scatter(transformedData[:,0], transformedData[:,1], c = colors2)
mpl.show()

# now try it with more topics
for ntopics in range(10)[1:]:
    ntopics
    nmf = NMF(n_components=ntopics, random_state=1).fit(tfidf)
    transformedData = nmf.fit_transform(tfidf)
    km = KMeans(2)
    correct = np.repeat([0,1], [len(techList), len(healthList)])
    guess = km.fit_predict(transformedData)
    np.mean(guess == correct)

# note that accuracy after 1 topics howevers around 75%
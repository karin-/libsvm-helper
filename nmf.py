from __future__ import print_function
#from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups

import re, urllib2, csv, time, threading, Queue

def getSummaries(stockList):
	"""returns summaries from given stock list"""
	text = []
	urls=[]
	for stock in stockList:
		urls.append(("http://finance.yahoo.com/q/pr?s=" + stock,))
	pages = run_parallel_in_threads(fetch, urls)
	while not pages.empty():
		html = pages.get()
		counter = 0
		pStart = 0
		while counter < 2:
			pStart += html[pStart:].find("Business Summary")
			counter += 1
		pEnd = html.find("</p>", pStart)
		text += [html[pStart + 73 : pEnd]]
	return text

# utility - spawn a thread to execute target for each args
def run_parallel_in_threads(target, args_list):
	"""helper function for getSummaries to parallelize page lookup"""
	result = Queue.Queue()
	# wrapper to collect return value in a Queue
	def task_wrapper(*args):
		result.put(target(*args))
	threads = [threading.Thread(target=task_wrapper, args=args) for args in args_list]
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	return result

def dummy_task(n):
	for i in xrange(n):
		time.sleep(0.1)
	return n

def fetch(url):
	"""returns html page of given url"""
	return urllib2.urlopen(url).read()


def getStockList(filename):
	"""reads stock names from csv file and returns them as list"""
	stockList = []
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			stockList += [row[0].strip()]
	return stockList

def printSummaries(summaryList, filename):
	output = open(filename, 'w')
	for summary in summaryList:
		output.write(summary + "\n")
	output.close()

# define global vars
techList = getStockList('companylist.csv')[1:]
print("created techList")
techSummaries = getSummaries(techList)
printSummaries(techSummaries, "techSummaries.txt")
print("retrieved tech summaries")
healthList = getStockList('companylisthealth.csv')[1:]
print("created healthList")
healthSummaries = getSummaries(healthList)
print("retrieved health summaries")
vocab = set()
vocabList = []


n_samples = 2000
n_features = 1000
n_topics = 2
n_top_words = 20

dataset = techSummaries + healthSummaries

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                             stop_words='english')
#tfidf = vectorizer.fit_transform(dataset.data[:n_samples])
tfidf = vectorizer.fit_transform(dataset)
#print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
#print("done in %0.3fs." % (time() - t0))

feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
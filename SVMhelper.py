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

# define global vars
techList = getStockList('companylist.csv')[1:]
print "created techList"
techSummaries = getSummaries(techList)
print "retrieved tech summaries"
healthList = getStockList('companylisthealth.csv')[1:]
print "created healthList"
healthSummaries = getSummaries(healthList)
print "retrieved health summaries"
vocab = set()
vocabList = []

def words(text):
	"""returns all words found in given string"""
	return re.findall('[a-z]+', text.lower()) 

def createVectors(summaryLst):
	"""returns list of bag-of-words vectors for each doc in given list"""
	global vocab
	global vocabList
	global count
	allVectors=[]
	for document in summaryLst:
		totalWordArr = words(document)
		currentDict = {}
		for word in totalWordArr:
			if not word in vocab:
				vocab.add(word)
				vocabList.append(word)
				currentDict[word] = 1
			else:
				k = 0
				if word in currentDict:
					k = currentDict[word]
				currentDict[word] = k + 1
		allVectors.append(currentDict)
	return allVectors

#define global vars
techVectors = createVectors(techSummaries)
healthVectors = createVectors(healthSummaries)

def printVectors(vectorList1, vectorList2, filename):
	"""writes vectors into output file suitable for libsvm"""
	output = open(filename, 'w')
	for vector in vectorList1:
		curr = "0"
		for i in range(len(vocabList)):
			k = 0
			if vocabList[i] in vector:
				k = vector[vocabList[i]]
			curr += " " + str(i+1)+":"+str(k)
		output.write(curr + "\n")
	for vector in vectorList2:
		curr = "1"
		for i in range(len(vocabList)):
			k = 0
			if vocabList[i] in vector:
				k = vector[vocabList[i]]
			curr += " " + str(i+1)+":"+str(k)
		output.write(curr + "\n")
	output.close()

printVectors(techVectors, healthVectors, '/Users/mdelangis/Desktop/libsvm-master/input.txt')











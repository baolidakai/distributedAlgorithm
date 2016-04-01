'''
PySpark implementation of all-prefix-sum algorithm
A' = sums of adjacent pairs, in parallel
R' = All-prefix-sum(A')
Fill in missing entries of R' using another n/2 processors, in parallel
'''
from pyspark import SparkContext
import random
sc = SparkContext()
# Compute the sums of adjacent pairs
# Read the input data
inputFilename = 'arr.txt'
# Construct the data by myself
N = 10
with open(inputFilename, 'w') as f:
	for i in range(N):
		f.write(str(random.randint(0, 100)))
		f.write('\n')
	f.close()
arr = sc.textFile(inputFilename).zipWithIndex().map(lambda x: (x[1], int(x[0])))
def cumSum(arr):
	'''
	Input:
	An RDD with [(idx, value)] containing the array to compute all prefix sums
	'''
	origSize = arr.map(lambda x: x[0]).reduce(lambda x, y: max(x, y)) + 1
	print('Computing cumulative sum for array of size %s' % origSize)
	if origSize <= 1:
		return arr
	# Pass both elements at 2i and 2i + 1 to the same key i
	sumAdj = arr.map(lambda x: (x[0] // 2, x[1])).reduceByKey(lambda x, y: x + y)
	# Recursively call all prefix sum to sumAdj
	'''
	# Use brute force first: map (idx, value) to (0, value), (1, value), ..., (idx, value)
	# Compute the size of sumAdj
	size = sumAdj.map(lambda x: x[0]).reduce(lambda x, y: max(x, y)) + 1
	recursiveResult = sumAdj.flatMap(lambda x: [(i, x[1]) for i in range(x[0], size)]).reduceByKey(lambda x, y: x + y)
	'''
	recursiveResult = cumSum(sumAdj)
	# Fill in the known values with recursiveResult
	known = recursiveResult.map(lambda x: (x[0] * 2 + 1, x[1]))
	# Fill in missing entries of R'
	# Let rtn emit (idx + 1, value), arr emit (idx, value) if idx is even
	rtn = known.flatMap(lambda x: [(x[0] + 1, x[1])] if x[0] + 1 < origSize else []).union(arr.flatMap(lambda x: [x] if x[0] % 2 == 0 else [])).reduceByKey(lambda x, y: x + y).union(known)
	print('Finished computing cumulative sum for array of size %s' % origSize)
	return rtn
res = cumSum(arr).sortByKey().values().collect()
sc.stop()
# Compare the result with the ground truth
data = []
with open(inputFilename, 'r') as f:
	for line in f:
		data.append(int(line))
	f.close()
groundTruth = [0] * len(data)
for i in range(len(data)):
	if i == 0:
		groundTruth[i] = data[i]
	else:
		groundTruth[i] = groundTruth[i - 1] + data[i]
for i in range(len(data)):
	if res[i] != groundTruth[i]:
		raise Exception('Result not matching!')

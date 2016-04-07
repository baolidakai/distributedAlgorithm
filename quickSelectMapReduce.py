'''
Parallel implementation of randomized selection algorithm
Select the k-th largest element in an array
p = random pivot
L = elements < p
R = elements > p
Recurse on L or R
'''
from pyspark import SparkContext
import numpy as np
sc = SparkContext()

# Construct the data
N = 8
data = sc.parallelize([np.random.rand() for i in range(N)]).zipWithIndex().map(lambda x: (x[1], x[0])).cache()
k = np.random.randint(N)

# Compute cumulated sum, borrowed from allPrefixSumMapReduce
def cumSum(arr):
	'''
	Input:
	An RDD with [(idx, value)] containing the array to compute all prefix sums
	'''
	origSize = arr.map(lambda x: x[0]).reduce(lambda x, y: max(x, y)) + 1
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
	rtn = known.filter(lambda x: x[0] + 1 < origSize).map(lambda x: (x[0] + 1, x[1])).union(arr.filter(lambda x: x[0] % 2 == 0)).reduceByKey(lambda x, y: x + y).union(known)
	return rtn

def naiveSelect(rdd, k):
	# Select the k-th element naively
	# Sort the rdd
	return rdd.map(lambda x: (x[1], x[0])).sortByKey().map(lambda x: x[0]).zipWithIndex().map(lambda x: (x[1], x[0])).filter(lambda x: x[0] == k - 1).values().take(1)[0]

def quickSelect(rdd, k):
	size = rdd.map(lambda x: x[0]).reduce(max) + 1
	print('quickSelect(%s)' % size)
	if size == 1:
		return rdd.values().collect()[0]
	# Randomly choose a pivot
	randomIdx = np.random.randint(size)
	pivot = rdd.filter(lambda x: x[0] == randomIdx).values().take(1)[0]
	# Bitmap indicating whether a number is strictly less than p
	leftBitmap = rdd.map(lambda x: (x[0], 1 if x[1] < pivot else 0))
	# Compute the cumulated sum
	leftCumsum = cumSum(leftBitmap).sortByKey()
	leftCumsum.persist()
	# Get the indices of the left elements
	leftIdx = leftBitmap.filter(lambda x: x[1] == 1).join(leftCumsum).map(lambda x: (x[0], x[1][1] - 1))
	# Construct L
	L = leftIdx.join(rdd).values()
	leftSize = leftBitmap.values().sum()
	if leftSize == k - 1:
		return(pivot)
	elif leftSize > k - 1:
		return(quickSelect(L, k))
	else:
		rightBitmap = rdd.map(lambda x: (x[0], 1 if x[1] > pivot else 0))
		rightCumsum = cumSum(rightBitmap).sortByKey()
		rightIdx = rightBitmap.filter(lambda x: x[1] == 1).join(rightCumsum).map(lambda x: (x[0], x[1][1] - 1))
		R = rightIdx.join(rdd).values()
		return(quickSelect(R, k - leftSize - 1))


groundTruth = naiveSelect(data, k)
res = quickSelect(data, k)
print('Ground truth is %s, quick selection gives %s' % (groundTruth, res))

sc.stop()

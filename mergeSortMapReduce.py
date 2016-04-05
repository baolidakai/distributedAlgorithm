'''
Parallel implementation of MergeSort algorithm
Total work: O(nlog^2(n)), depth: O(log^2(n))
Split the array into two parts, sort them recursively,
to merge these two parts: Compute the rank of L and R for each element, directly emit (idx, value) pair
Assume all elements are distinct
'''
from pyspark import SparkContext
import random

sc = SparkContext()
f = open('arr.txt', 'w')
N = 100
data = [random.random() for i in range(N)]
rdd = sc.parallelize(data).zipWithIndex().map(lambda x: (x[1], float(x[0])))

def mergeSort(rdd):
	N = rdd.keys().reduce(max) + 1
	print('Calling on an array of size %s' % N)
	if N <= 1:
		return rdd
	left = rdd.filter(lambda x: x[0] < N // 2)
	right = rdd.filter(lambda x: x[0] >= N // 2).map(lambda x: (x[0] - N // 2, x[1]))
	sortedLeft = mergeSort(left).values().collect()
	sortedRight = mergeSort(right).values().collect()
	N1 = len(sortedLeft)
	N2 = len(sortedRight)
	# Let each element in sortedLeft emit its rank in the merged array: ranks are 0, 1, ..., N - 1
	def smallerInArray(val, arr):
		# Use binary search to find the number of elements < val in arr
		if val <= arr[0]:
			return 0
		if val > arr[-1]:
			return len(arr)
		lo = 0
		hi = len(arr) - 1
		while lo < (lo + hi) // 2:
			mi = (lo + hi) // 2
			if arr[mi] >= val:
				hi = mi
			else:
				lo = mi
		return lo + 1
	def leftFinalRank(indexLeft):
		# Final rank is the index in the left + number of elements < curr value in the right
		return indexLeft + smallerInArray(sortedLeft[indexLeft], sortedRight)
	def rightFinalRank(indexRight):
		return indexRight + smallerInArray(sortedRight[indexRight], sortedLeft)
	# Merge the results
	part1 = sc.parallelize(sortedLeft).zipWithIndex().map(lambda x: (leftFinalRank(x[1]), x[0]))
	part2 = sc.parallelize(sortedRight).zipWithIndex().map(lambda x: (rightFinalRank(x[1]), x[0]))
	rtn = part1.union(part2).sortByKey()
	return rtn

res = mergeSort(rdd)
print(res.collect())
sc.stop()

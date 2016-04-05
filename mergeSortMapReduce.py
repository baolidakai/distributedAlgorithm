'''
Parallel implementation of MergeSort algorithm
Total work: O(nlog^2(n)), depth: O(log^2(n))
Split the array into two parts, sort them recursively,
to merge these two parts: Compute the rank of L and R for each element, directly emit (idx, value) pair
'''
from pyspark import SparkContext

sc = SparkContext()
rdd = sc.textFile('arr.txt').zipWithIndex().map(lambda x: (x[1], float(x[0]))).cache()

def naiveSort(rdd):
	# Sort an array using the naive selection sort
	vals = rdd.

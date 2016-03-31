'''
Multithreading implementation of all-prefix-sum algorithm
A' = sums of adjacent pairs, in parallel
R' = All-prefix-sum(A')
Fill in missing entries of R' using another n/2 processors, in parallel
'''
import threading
from queue import Queue
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def allPrefixSum(arr, p = 10):
	'''
	Implements the all prefix sum algorithm recursively, in parallel
	Input:
	arr - the array to compute
	'''
	n = len(arr)
	if n <= 1:
		return arr[:]
	# Compute A', the sum of adjacent pairs, in parallel
	adjSum = [0] * (n // 2)
	def computePairSum(idx):
		'''
		Compute the sum of arr[idx * 2] and arr[idx * 2 + 1], and assign to A'[idx]
		'''
		res = arr[idx * 2] + arr[idx * 2 + 1]
		adjSum[idx] = res
	def threader():
		while True:
			worker = q.get()
			computePairSum(worker)
			q.task_done()
	q = Queue()
	for _ in range(p):
		t = threading.Thread(target = threader)
		t.daemon = True
		t.start()
	for worker in range(n // 2):
		q.put(worker)
	q.join()
	# Recursively compute R'
	everyOtherSum = allPrefixSum(adjSum)
	rtn = [0] * n
	# Fill in every other sum in the true result
	rtn[1::2] = everyOtherSum
	# Fill in the rest elements, in parallel
	def fillInRest(idx):
		'''
		Compute the sum of arr[idx * 2] and arr[idx * 2 + 1], and assign to A'[idx]
		'''
		res = arr[idx * 2]
		if idx > 0:
			res += rtn[idx * 2 - 1]
		rtn[idx * 2] = res
	def threader():
		while True:
			fillInRest(q.get())
			q.task_done()
	q = Queue()
	for _ in range(p):
		t = threading.Thread(target = threader)
		t.daemon = True
		t.start()
	for worker in range(n - n // 2):
		q.put(worker)
	q.join()
	return rtn

# Runtime visualization
Ns = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
parallelTimes = np.zeros_like(Ns)
naiveTimes = np.zeros_like(Ns)
for i, N in enumerate(Ns):
	arr = [random.randint(0, 100) for _ in range(N)]
	timeStart = time.time()
	_ = allPrefixSum(arr)
	timeEnd = time.time()
	parallelTimes[i] = (timeEnd - timeStart) * 1000000
	timeStart = time.time()
	res = list(range(N))
	for j in range(N):
		if j == 0:
			res[j] = arr[j]
		else:
			res[j] = res[j - 1] + arr[j]
	timeEnd = time.time()
	naiveTimes[i] = (timeEnd - timeStart) * 1000000

plt.plot(Ns, naiveTimes, label = 'Naive algorithm')
plt.plot(Ns, parallelTimes, label = 'Parallel algorithm')
plt.xlabel('N')
plt.ylabel('Runtime in ms / 1000')
plt.legend(loc = 'upper center')
plt.show()

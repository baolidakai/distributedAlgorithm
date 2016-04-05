'''
Implements the Strassen algorithm in parallel
'''
from pyspark import SparkContext
import sys
sc = SparkContext()

def textToMatrix(rdd): # Convert a matrix in text file to ((i, j), m[i][j])
	return rdd.zipWithIndex().flatMap(lambda x: [((x[1], j), float(value)) for j, value in enumerate(x[0].split(' '))])
A = textToMatrix(sc.textFile('matrix1.txt')).cache()
B = textToMatrix(sc.textFile('matrix2.txt')).cache()

def shape(m): # Return the size of m
	return m.map(lambda x: x[0][0]).reduce(max) + 1

def multiply(m1, m2): # Computing the multiplication of m1 and m2
	'''
	((i, j, k), mij))
	((i, j, k), njk)
	=> ((i, j, k), mij * njk)
	'''
	# Compute the dimension of m1 and m2
	N = shape(m1)
	# Emit ((i, j, k), mij) pairs
	output1 = m1.flatMap(lambda elem: [((elem[0][0], elem[0][1], k), elem[1]) for k in range(N)]).cache()
	# Emit ((i, j, k), njk) pairs
	output2 = m2.flatMap(lambda elem: [((i, elem[0][0], elem[0][1]), elem[1]) for i in range(N)]).cache()
	# Join the results
	joined = output1.join(output2).map(lambda x: (x[0], x[1][0] * x[1][1])).cache()
	rtn = joined.map(lambda x: ((x[0][0], x[0][2]), x[1])).reduceByKey(lambda x, y: x + y).cache()
	return rtn

def offset(m, rowOffset, colOffset): # Reduce all row index by rowOffset, reduce all col index by colOffset
	return m.map(lambda x: ((x[0][0] - rowOffset, x[0][1] - colOffset), x[1])).cache()

def add(m1, m2): # Add two matrices m1 and m2
	return m1.join(m2).map(lambda x: (x[0], x[1][0] + x[1][1])).cache()

def subtract(m1, m2): # Subtract two matrices m1 and m2
	return m1.join(m2).map(lambda x: (x[0], x[1][0] - x[1][1])).cache()

def strassenMultiply(A, B):
	# Only work for A: 2^n x 2^n, B: 2^n x 2^n
	N = shape(A)
	if N <= 1:
		return A.join(B).map(lambda x: (x[0], x[1][0] * x[1][1])).cache()
	A11 = A.filter(lambda x: x[0][0] < N // 2 and x[0][1] < N // 2).cache()
	A12 = offset(A.filter(lambda x: x[0][0] < N // 2 and x[0][1] >= N // 2), 0, N / 2).cache()
	A21 = offset(A.filter(lambda x: x[0][0] >= N // 2 and x[0][1] < N / 2), N // 2, 0).cache()
	A22 = offset(A.filter(lambda x: x[0][0] >= N // 2 and x[0][1] >= N // 2), N // 2, N // 2).cache()
	B11 = B.filter(lambda x: x[0][0] < N // 2 and x[0][1] < N // 2).cache()
	B12 = offset(B.filter(lambda x: x[0][0] < N // 2 and x[0][1] >= N // 2), 0, N // 2).cache()
	B21 = offset(B.filter(lambda x: x[0][0] >= N // 2 and x[0][1] < N // 2), N // 2, 0).cache()
	B22 = offset(B.filter(lambda x: x[0][0] >= N // 2 and x[0][1] >= N // 2), N // 2, N // 2).cache()
	'''
	M1 = (A11 + A22)(B11 + B22)
	M2 = (A21 + A22)B11
	M3 = A11(B12 - B22)
	M4 = A22(B21 - B11)
	M5 = (A11 + A12)B22
	M6 = (A21 - A11)(B11 + B12)
	M7 = (A12 - A22)(B21 + B22)
	'''
	M1 = strassenMultiply(add(A11, A22), add(B11, B22)).cache()
	M2 = strassenMultiply(add(A21, A22), B11).cache()
	M3 = strassenMultiply(A11, subtract(B12, B22)).cache()
	M4 = strassenMultiply(A22, subtract(B21, B11)).cache()
	M5 = strassenMultiply(add(A11, A12), B22).cache()
	M6 = strassenMultiply(subtract(A21, A11), add(B11, B12)).cache()
	M7 = strassenMultiply(subtract(A12, A22), add(B21, B22)).cache()
	'''
	C11 = M1 + M4 - M5 + M7
	C12 = M3 + M5
	C21 = M2 + M4
	C22 = M1 - M2 + M3 + M6
	'''
	C11 = add(subtract(add(M1, M4), M5), M7).cache()
	C12 = add(M3, M5).cache()
	C21 = add(M2, M4).cache()
	C22 = add(add(subtract(M1, M2), M3), M6).cache()
	# Join the results
	rtn = C11.union(offset(C12, 0, -N//2)).union(offset(C21, -N//2, 0)).union(offset(C22, -N//2, -N//2)).cache()
	return rtn

def strassenMultiplyGeneral(A, B):
	# A: n1 x n2
	# B: n2 x n3
	n1 = A.map(lambda x: x[0][0]).reduce(max) + 1
	n2 = A.map(lambda x: x[0][1]).reduce(max) + 1
	n3 = B.map(lambda x: x[0][1]).reduce(max) + 1
	logSize = 0
	size = 1
	while size < max(n1, n2, n3):
		logSize += 1
		size *= 2
	# Pad with 0s to make both matrices of size size
	# Expand each row of A
	rowExtendA = A.flatMap(lambda x: [x] + [((x[0][0], j), 0.) for j in range(n2, size)] if x[0][1] == 0 else [x]).cache()
	# Expand each col of A
	colExtendA = rowExtendA.flatMap(lambda x: [x] + [((i, x[0][1]), 0.) for i in range(n1, size)] if x[0][0] == 0 else [x]).cache()
	# Expand each row of B
	rowExtendB = B.flatMap(lambda x: [x] + [((x[0][0], j), 0.) for j in range(n3, size)] if x[0][1] == 0 else [x]).cache()
	# Expand each col of B
	colExtendB = rowExtendB.flatMap(lambda x: [x] + [((i, x[0][1]), 0.) for i in range(n2, size)] if x[0][0] == 0 else [x]).cache()
	rtn = strassenMultiply(colExtendA, colExtendB).cache()
	# Filter to the correct size
	return rtn.filter(lambda x: x[0][0] < n1 and x[0][1] < n3)

C = strassenMultiplyGeneral(A, B).cache()

#sc.stop()

import tvm
import tvm.testing
from tvm import te
import numpy as np
import timeit
import numpy

def evaluate_operation(s, vars, target, name, optimization, log):
    func = tvm.build(s, [A, B, C], target=target, name="matMult")
    assert func

    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))
    log.append((optimization, mean_time))

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 1024
K = 1024
N = 1024

# The default tensor data type in tvm
dtype = "float32"

target = tvm.target.Target(target="llvm", host="llvm")
dev = tvm.device(target.kind.name, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))

answer = numpy.dot(a.numpy(), b.numpy())

print("*********************************************************")

k = te.reduce_axis((0,K),"k_red")
A = te.placeholder((M,K),name="A")
B = te.placeholder((K,N),name="B")
C = te.compute((M,N),lambda x,y:te.sum(A[x,k]*B[k,y],axis= k),name="C")
#Schedule
s = te.create_schedule(C.op)
func = tvm.build(s,[A, B, C],target=target,name="matMult")

c = tvm.nd.array(np.zeros((M,N),dtype=dtype),dev)
func(a,b,c)
tvm.testing.assert_allclose(c.numpy(),answer, rtol=1e-5)

print(str(dev))
log = []
evaluate_operation(s, [A, B, C], target=target, name="matMult", optimization="none", log=log)
print("*********************************************************")

#Optimization 1: Blocking 
bn = 32

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(yo, xo, ko, ki, yi, xi)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="blocking", log=log)

print("*********************************************************")
#Optimization 2: Vectorization
#Add vector optimization to schedule above
s[C].vectorize(xi)
evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="vectorization", log=log)
print("*********************************************************")
#Optimization 3: Loop Permutation 

print("*********************************************************")
#Optimization 4: Array Packing

print("*********************************************************")
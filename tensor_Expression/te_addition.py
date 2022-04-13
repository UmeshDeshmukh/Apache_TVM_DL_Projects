import tvm
import tvm.testing
from tvm import te
import numpy as np
import timeit

tgt = tvm.target.Target(target="llvm -mcpu=kabylake-avx512",host="llvm")

# shape n
n = te.var("n")
# tensor A with shape n
A = te.placeholder((n,), name="A")
# tensor B with shape n
B = te.placeholder((n,), name="B")

C = te.compute(A.shape,lambda i:A[i]+B[i],name="C")

# Schedule for above computation
s = te.create_schedule(C.op)
custom_add1 = tvm.build(s,[A,B,C],tgt,name='add_func')

dev = tvm.device(tgt.kind.name, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
custom_add1(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

################################################################################
np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)

#print("Numpy running time: %f" % (np_running_time / np_repeat))


def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))


log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(custom_add1, tgt, "naive", log=log)

###############################################################################

# New schedule to parallelize operation

s[C].parallel(C.op.axis[0])
#print(tvm.lower(s, [A, B, C], simple_mode=True))
add_parallel = tvm.build(s, [A, B, C], tgt, name="add_func_parallel")
add_parallel(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
evaluate_addition(add_parallel, tgt, "parallel", log=log)
###############################################################################

#this factor indicate no. of threads. Equal to no. of CPU cores.
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

split_factor = 4
outer,inner = s[C].split(C.op.axis[0],factor= split_factor)
s[C].parallel(outer)
s[C].vectorize(inner)
add_parallel_vect = tvm.build(s, [A, B, C], tgt, name="add_parallel_vect")
evaluate_addition(add_parallel_vect, tgt, "add_parallel_vect", log=log)

baseline = log[0][1]
print("%s\t%s\t%s" % ("Operator".rjust(20), "Timing".rjust(20), "Performance".rjust(20)))
for result in log:
    print(
        "%s\t%s\t%s"
        % (result[0].rjust(20), str(result[1]).rjust(20), str(result[1] / baseline).rjust(20))
    )

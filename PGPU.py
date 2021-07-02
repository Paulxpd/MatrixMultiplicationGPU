from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import time
from numba import jit

N = 100

@cuda.jit
def create_matrix(rng_states, out1, out2):
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(rng_states, thread_id)
    y = xoroshiro128p_uniform_float32(rng_states, thread_id)
    out1[thread_id] = (int)(x * 100)
    out2[thread_id] = (int)(y * 100)

rng_states = create_xoroshiro128p_states(N * N, seed = 1)
out1 = np.zeros(N * N, dtype = np.float32)
out2 = np.zeros(N * N, dtype = np.float32)

create_matrix[N,N](rng_states, out1, out2)
x = out1.reshape((N,N))
y = out2.reshape((N,N))
zcpu = np.zeros((N,N))
zgpu = np.zeros((N,N))
z = np.zeros((N,N))

def cpu(x,y,z,N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                z[i-1][j-1] += x[i-1][k-1] * y[k-1][j-1]
    return z

@jit(nopython=True)
def gpu(x,y,z,N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                z[i-1][j-1] = x[i-1][k-1] * y[k-1][j-1]
    return z

cpu(x,y,z,2) #compiling function
gpu(x,y,z,2)
start = time.time()
print("CPU (Single Core): ")
cpu_result = cpu(x,y,zcpu,N)
print(cpu_result)
end = time.time()
print("CPU Time = %s" % (end - start))
start = time.time()
print("GPU: ")
gpu_result = gpu(x,y,zgpu,N)
print(gpu_result)
end = time.time()
print("GPU Time = %s" % (end-start))

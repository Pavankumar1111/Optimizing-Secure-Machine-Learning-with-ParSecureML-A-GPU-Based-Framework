import numpy as np
import multiprocessing as mp
import time
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import matplotlib.pyplot as plt

# Initialize CPU parallel processing for matrix multiplication
def cpu_parallel_matrix_multiply(A, B, result, segment):
    """Perform segment of matrix multiplication in parallel."""
    for i in range(segment[0], segment[1]):
        for j in range(B.shape[1]):
            sum = 0
            for k in range(A.shape[1]):
                sum += A[i][k] * B[k][j]
            result[i][j] = sum

def gpu_matrix_multiply(A, B):
    """Perform matrix multiplication using GPU."""
    kernel_code = """
   __global__ void MatrixMulKernel(float *a, float *b, float *c, int WIDTH)
{
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;
    for (int i = 0; i < WIDTH; i++) {
        tmpSum += a[ROW * WIDTH + i] * b[i * WIDTH + COL];
    }
    c[ROW * WIDTH + COL] = tmpSum;
}

    """

    kernel_module = compiler.SourceModule(kernel_code)
    matrixmul = kernel_module.get_function("MatrixMulKernel")

    a_gpu = gpuarray.to_gpu(A.astype(np.float32))
    b_gpu = gpuarray.to_gpu(B.astype(np.float32))
    c_gpu = gpuarray.empty((A.shape[0], B.shape[1]), np.float32)

    thread_size = (16, 16, 1)
    grid_size = (int(np.ceil(B.shape[1] / 16)), int(np.ceil(A.shape[0] / 16)))

    matrixmul(a_gpu, b_gpu, c_gpu, np.int32(A.shape[1]), block=thread_size, grid=grid_size)

    return c_gpu.get()

# Example matrix dimensions
N = 1024
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# CPU parallel execution
cpu_result = np.zeros((N, N))
num_processes = mp.cpu_count()
processes = []
segment_size = N // num_processes

start_time = time.time()
for i in range(num_processes):
    start = i * segment_size
    end = (i + 1) * segment_size if i < num_processes - 1 else N
    segment = (start, end)
    p = mp.Process(target=cpu_parallel_matrix_multiply, args=(A, B, cpu_result, segment))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

cpu_execution_time = time.time() - start_time
print(f"CPU Parallel Execution Time: {cpu_execution_time} seconds")

# GPU execution
start_time = time.time()
gpu_result = gpu_matrix_multiply(A, B)
gpu_execution_time = time.time() - start_time
print(f"GPU Execution Time: {gpu_execution_time} seconds")

# Plotting
tasks = ['CPU Parallel', 'GPU']
performance = [cpu_execution_time, gpu_execution_time]

plt.figure(figsize=(8, 6))
plt.bar(tasks, performance, color=['blue', 'red'])
plt.xlabel('Tasks')
plt.ylabel('Execution Time (seconds)')
plt.title('Comparison of CPU Parallel and GPU Execution Times')
plt.show()
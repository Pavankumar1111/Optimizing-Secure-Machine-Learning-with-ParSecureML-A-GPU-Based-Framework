import numpy as np
import threading
from numba import cuda, float32
import matplotlib.pyplot as plt

# Thread-safe random number generation function
def thread_safe_random(start, end):
    rng = np.random.default_rng()  # Each thread gets its own RNG instance
    return rng.integers(start, end)

def worker():
    # Example worker function to demonstrate thread-safe RNG
    print(f"Thread-safe random number: {thread_safe_random(1, 100)}")

# Efficient matrix operations using numpy
def efficient_matrix_operations():
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    # Matrix addition and subtraction
    addition_result = A + B
    subtraction_result = A - B
    print("Matrix addition and subtraction performed efficiently.")

# GPU-accelerated matrix addition using Numba
@cuda.jit
def add_matrices_gpu(A, B, result):
    x, y = cuda.grid(2)
    if x < result.shape[0] and y < result.shape[1]:
        result[x, y] = A[x, y] + B[x, y]

def gpu_matrix_addition():
    A = np.random.rand(1000, 1000).astype(np.float32)
    B = np.random.rand(1000, 1000).astype(np.float32)
    result = np.zeros_like(A)
    # Define grid size for GPU execution
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # Perform the matrix addition on GPU
    add_matrices_gpu[blockspergrid, threadsperblock](A, B, result)
    print("Matrix addition performed on GPU.")

# Main execution block to run the examples
if __name__ == "__main__":
    # Demonstrate thread-safe RNG with 5 worker threads
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Perform efficient matrix operations
    efficient_matrix_operations()

    # Demonstrate GPU matrix addition if CUDA is available
    if cuda.is_available():
        gpu_matrix_addition()
    else:
        print("CUDA is not available. GPU matrix addition skipped.")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Example data (Replace with actual data)
    tasks = ['Thread-safe RNG', 'Matrix Operations (CPU)', 'Matrix Addition (GPU)']
    performance = [5, 1, 1]  # Placeholder values for demonstration

    # Creating the bar plot
    plt.bar(tasks, performance, color=['blue', 'orange', 'green'])

    # Adding titles and labels
    plt.xlabel('Tasks')
    plt.ylabel('Performance (seconds)')
    plt.title('Performance Comparison of Different Tasks')
    plt.ylim(0, 6)  # Adjust the y-axis limits according to your actual performance values

    # Displaying the plot
    plt.show()

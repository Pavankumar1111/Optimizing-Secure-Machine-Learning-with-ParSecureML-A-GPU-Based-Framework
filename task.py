import time
import random
from concurrent.futures import ThreadPoolExecutor

# Mock functions to simulate ML tasks' workload
def simulate_cnn(dataset):
    time.sleep(random.uniform(0.5, 1.5))  # Simulating variable computation time

def simulate_mlp(dataset):
    time.sleep(random.uniform(0.2, 0.7))

def simulate_rnn(dataset):
    time.sleep(random.uniform(0.4, 1.0))

def simulate_linear_regression(dataset):
    time.sleep(random.uniform(0.1, 0.3))

def simulate_logistic_regression(dataset):
    time.sleep(random.uniform(0.2, 0.6))

def simulate_svm(dataset):
    time.sleep(random.uniform(0.3, 0.8))

# Function to simulate dataset loading
def load_dataset(name):
    # Mock dataset loading time
    time.sleep(random.uniform(0.1, 0.2))
    return f"{name}_data"

# Performance measurement simulation
def measure_performance(task_func, dataset, parallel=False):
    start_time = time.time()
    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(task_func, dataset)
    else:
        task_func(dataset)
    end_time = time.time()
    return end_time - start_time

# Example usage
datasets = ['VGGFace2', 'NIST', 'CIFAR-10', 'SYNTHETIC', 'MNIST']
ml_tasks = [simulate_cnn, simulate_mlp, simulate_rnn, simulate_linear_regression, simulate_logistic_regression, simulate_svm]

for dataset in datasets:
    data = load_dataset(dataset)
    for task in ml_tasks:
        performance = measure_performance(task, data, parallel=True if dataset == 'SYNTHETIC' else False)
        print(f"Task: {task.__name__}, Dataset: {dataset}, Performance: {performance:.2f} seconds")

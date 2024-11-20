import random
import time
import numpy as np
import pandas as pd  # For the table
import matplotlib.pyplot as plt  # For plotting
import minitorch

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    sizes = [64, 128, 256, 512, 1024]
    times = []

    for size in sizes:
        print(f"Running size {size}")
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            # Fast backend timing
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()
            fast_times.append(end_fast - start_fast)

            # GPU backend timing
            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()
            gpu_times.append(end_gpu - start_gpu)

        # Store average times
        avg_fast_time = np.mean(fast_times)
        avg_gpu_time = np.mean(gpu_times)
        times.append({"Size": size, "Backend": "Fast", "Time": avg_fast_time})
        times.append({"Size": size, "Backend": "GPU", "Time": avg_gpu_time})

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(times)
    print("\nTiming Summary Table:")
    print(df)

    # Plot results using only matplotlib
    plt.figure(figsize=(10, 6))
    for backend in df["Backend"].unique():
        backend_data = df[df["Backend"] == backend]
        plt.plot(
            backend_data["Size"],
            backend_data["Time"],
            marker="o",
            label=f"{backend} Backend"
        )

    plt.title("Matrix Multiplication Timing")
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("timing_plot.png")
    plt.show()

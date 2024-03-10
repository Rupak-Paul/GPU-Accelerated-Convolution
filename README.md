# GPU-Accelerated-Convolution
GPU Parallel algorithm to implement the convolution operation on a 2D matrix using a 2D filter. Various memory optimization such as memory coalescing, make use of Shared-Memory and Constant-Memory are used to gain maximum performance benefits.

Synchronization between threads has been taken care to maintain the correct state.

Achieved ~400x performance gain as compared to sequential CPU based execution.

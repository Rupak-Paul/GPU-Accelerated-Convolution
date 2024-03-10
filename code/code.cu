#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;
using std::cin;
using std::cout;


__global__
void convolution(long int* matrix, long int* filter, long int* ans, int m, int n, int k) {
    extern __shared__ int sharedFilter[];

    int stripSize = ceil((double)(k*k) / blockDim.x);
    for(int i = 0; i < stripSize; i++) {
        //For example, If k*k = 16 and blockDim.x = 5 then this is how memory access is being done by threads in a thread-block
        //indices of filter matrix: 0 1 2 3 4  5 6 7 8 9  10 11 12 13 14  15
        //accessed by threadIdx.x:  0 1 2 3 4  0 1 2 3 4   0  1  2  3  4   0 1 2 3 4
        int index = threadIdx.x + i*blockDim.x; //optimization for coalesced memeory access 
        if(index < k*k) {
            sharedFilter[index] = filter[index];
        }
    }

    __syncthreads();

    if(threadIdx.x < n) {
        int firstX = blockIdx.x - k/2;
        int lastX = blockIdx.x + k/2;
        int firstY = threadIdx.x - k/2;
        int lastY = threadIdx.x + k/2;

        long int sum = 0;
        for(int i = firstX; i <= lastX; i++) {
            for(int j = firstY; j <= lastY; j++) {
                if(i < 0 || i >= m) continue;
                if(j < 0 || j >= n) continue;

                int filterIdX = i - firstX;
                int filterIdY = j - firstY;

                sum += matrix[i*n + j] * sharedFilter[filterIdX*k + filterIdY];
            }
        }

        ans[blockIdx.x*n + threadIdx.x] = sum;
    }
}


int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];
    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }
    

    long int* d_mat;
    long int* d_filter;
    long int* d_ans;

    cudaMalloc(&d_mat, m*n*sizeof(long int));
    cudaMalloc(&d_filter, k*k*sizeof(long int));
    cudaMalloc(&d_ans, m*n*sizeof(long int));

    cudaMemcpy(d_mat, h_mat, m*n*sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, k*k*sizeof(long int), cudaMemcpyHostToDevice);

    int blockSize = (k*k > n ? k*k : n);
    if(blockSize > 1024) blockSize = 1024;

    auto start = std::chrono::high_resolution_clock::now();
    
    convolution<<<m, blockSize, k*k*sizeof(int)>>>(d_mat, d_filter, d_ans, m, n, k);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(h_ans, d_ans, m*n*sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_filter);
    cudaFree(d_ans);

    
    //printing output and timings (time to compute convolution) in a file
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
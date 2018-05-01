
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include </usr/local/cuda-8.0/targets/x86_64-linux/include/cuda_profiler_api.h>

// Set the tile width, use 16 for now since it is suggested in the textbook
#define TILE_WIDTH 16

// TEST: Try to declare the constant memory stored array for kernel
#define MAX_KERNEL_SIZE 2400
__constant__ float W[MAX_KERNEL_SIZE];


namespace mxnet
{
namespace op
{

// The kernel code
// __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {

// Modified version for running with the constant memory
__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int n, m, c, h, w, p, q;
    int h0, w0, h_base, w_base;
    int W_grid = (W_out-1) / TILE_WIDTH + 1;

    // Since each thread block will be responsible for computing one 16x16 tile in the output
    // We need to set some of the parameters
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.y;
    w0 = threadIdx.x;

    h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;

    h = h_base + h0;
    w = w_base + w0;

    float acc = 0.;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    //#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Modified version for running with the constant memory
    #define k4d(i3, i2, i1, i0) W[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    if ((h < H_out) && (w < W_out)){

      for (c = 0; c < C; c++) {
        for (p = 0; p < K; p++){
          for (q = 0; q < K; q++){
            acc += (x4d(n, c, h + p, w + q) * k4d(m, c, p, q));
          }
        }
      }
      y4d(n, m, h, w) = acc;
    }
    #undef y4d
    #undef x4d
    #undef k4d
}

/*
  IN PROGRESS: Finishing up the code and wait for modification
*/
// __global__ void unroll_Kernel (int C, int H, int W, int K, float* X, float X_unroll){
//   int c, s, p, q;
//   int h_out, w_out, h_unroll, w_unroll, w_base;
//   //@@ Notice: Need to define the value of MAX_NUM_THREADS
//   int t = blockIdx.x * MAX_NUM_THREADS + threadIdx.x;
//   int H_out = H - K + 1;
//   int W_out = W - K + 1;
//   int W_unroll = H_out * W_out;
//   #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]
//
//   if (t < C * W_unroll){
//     c = t / W_unroll;
//     s = t % W_unroll;
//     h_out = s / W_out;
//     w_out = s % W_out;
//     h_unroll = h_out * W_out + w_out;
//     w_base = c * K * K;
//     for (p = 0; p < K, p++){
//       for (q = 0; q < K, q++){
//         w_unroll = w_base + p*K + q;
//         //@@ NOTICE: Still need to figure out why here is x3d
//         X_unroll[h_unroll + (w_unroll * W_unroll)] = x3d(c, h_out+p, w_out+q);
//       }
//     }
//   }
//   #undef x4d
// }

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int N = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[2];

    // Need to checkout the dimension of the input data size
    printf("The Value of N = %d\n", N);
    printf("The Value of M = %d\n", M);
    printf("The Value of C = %d\n", C);
    printf("The Value of H = %d\n", H);
    printf("The Value of W = %d\n", W);
    printf("The Value of K = %d\n", K);

    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out-1) / TILE_WIDTH + 1;
    const int H_grid = (H_out-1) / TILE_WIDTH + 1;
    const int Z = H_grid * W_grid;

    dim3 gridDim(N, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    cudaMemcpyToSymbol (W, w, M*C*K*K*sizeof(float));

    if (M == 6) cudaProfilerStart();
    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, N,M,C,H,W,K);

    // Modified version for running with the constant memory
    forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_, N,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    if(M == 6) cudaProfilerStop();

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif

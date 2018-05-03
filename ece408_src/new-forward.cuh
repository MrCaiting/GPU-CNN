//
// #ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
// #define MXNET_OPERATOR_NEW_FORWARD_CUH_
//
// #include <mxnet/base.h>
// //#include "cublas.h"
// #include </usr/local/cuda-8.0/targets/x86_64-linux/include/cuda_profiler_api.h>
//
// // Set the tile width, use 16 for now since it is suggested in the textbook
// #define TILE_WIDTH 16
// #define STREAM 4
// #define MAX_THREAD 1024
// // TEST: Try to declare the constant memory stored array for kernel
// #define MAX_KERNEL_SIZE 2400
// __constant__ float MASK[MAX_KERNEL_SIZE];
//
//
// namespace mxnet
// {
// namespace op
// {
// // tiled matrix multiplication
// __global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols){
//   __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
//   __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
//
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//
//   // int row = by*TILE_WIDTH + ty;
//   // int col = bx*TILE_WIDTH + tx;
//   int row = by*TILE_WIDTH + ty;
//   int col = bx*TILE_WIDTH + tx;
//
//   float pvalue = 0;
//
//   for (int n = 0; n < (numACols-1)/TILE_WIDTH+1 ; ++n){
//     if ((row < numARows) && (n*TILE_WIDTH + tx < numACols)){
//       subTileM[ty][tx] = MASK[row*numACols + n*TILE_WIDTH+tx];
//     }
//     else{
//       subTileM[ty][tx] = 0.0;
//     }
//     if (col < numBCols && n*TILE_WIDTH + ty < numBRows){
//       subTileN[ty][tx] = B[(n*TILE_WIDTH+ty)*numBCols + col];
//     }
//     else{
//       subTileN[ty][tx] = 0.0;
//     }
//     __syncthreads();
//     if (row < numARows && col < numBCols){
//       for (int k = 0; k < TILE_WIDTH; ++k){
//         pvalue += subTileM[ty][k] * subTileN[k][tx];
//       }
//     }
//     __syncthreads();
//   }
//
//   if (row < numARows && col < numBCols){
//     C[row*numCCols + col] = pvalue;
//   }
// }
//
// /*
//   IN PROGRESS: Finishing up the code and wait for modification
// */
// __global__ void unroll_Kernel (int C, int H, int W, int K, float* x, float *X_unroll){
//   int c, s, p, q;
//   int h_out, w_out, h_unroll, w_unroll, w_base;
//   //@@ Notice: Need to define the value of MAX_THREAD
//   int t = blockIdx.x * MAX_THREAD + threadIdx.x;
//   int H_out = H - K + 1;
//   int W_out = W - K + 1;
//   int W_unroll = H_out * W_out;
//   #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]
//   if (t < C * W_unroll){
//     c = t / W_unroll;
//     s = t % W_unroll;
//     h_out = s / W_out;
//     w_out = s % W_out;
//     h_unroll = h_out * W_out + w_out;
//     w_base = c * K * K;
//     for (p = 0; p < K; p++){
//       for (q = 0; q < K; q++){
//         w_unroll = w_base + p*K + q;
//         X_unroll[h_unroll + (w_unroll * W_unroll)] = x3d(c, h_out+p, w_out+q);
//       }
//     }
//   }
//   #undef x3d
// }
//
// /*
//    This function is called by new-inl.h
//    Any code you write should be executed by this function.
//    For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
// */
// template <>
// void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
// {
//     // Extract the tensor dimensions into B,M,C,H,W,K
//     // ...
//     const int N = x.shape_[0];
//     const int M = y.shape_[1];
//     const int C = x.shape_[1];
//     const int H = x.shape_[2];
//     const int W = x.shape_[3];
//     const int K = w.shape_[2];
//
//     // Need to checkout the dimension of the input data size
//     printf("The Value of N = %d\n", N);
//     printf("The Value of M = %d\n", M);
//     printf("The Value of C = %d\n", C);
//     printf("The Value of H = %d\n", H);
//     printf("The Value of W = %d\n", W);
//     printf("The Value of K = %d\n", K);
//
//     // Set the kernel dimensions
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//     //const int W_grid = (W_out-1) / TILE_WIDTH + 1;
//     //const int H_grid = (H_out-1) / TILE_WIDTH + 1;
//     //const int Z = H_grid * W_grid;
//
//     int numXCols = H_out * W_out;
//     int numXRows = C * K * K;
//     int numWCols = numXRows;
//     int numWRows = M;
//     int numYCols = numXCols;
//     int numYRows = numWRows;
//
//
//     cudaStream_t curr_stream;
//     cudaStream_t streams_array[STREAM];
//     int streamIdx;
//     float * x_unroll;
//
//     for (int i = 0; i < STREAM; ++i) {
//       cudaStreamCreate(&streams_array[i]);
//     }
//
//     cudaMalloc((void**) &x_unroll, H_out * W_out * C * K * K * sizeof(float)*STREAM);
//     cudaMemcpyToSymbol (MASK, w.dptr_, M*C*K*K*sizeof(float));
//
//     dim3 gridDim_unroll(ceil(float(C * H_out * W_out)/ float(MAX_THREAD)));
//
//     dim3 gridDim_mul((numYCols - 1)/TILE_WIDTH + 1, (numYRows-1)/TILE_WIDTH + 1, 1);
//     dim3 blockDim_mul(TILE_WIDTH, TILE_WIDTH, 1);
//     //if (M == 6) cudaProfilerStart();
//     for (int n = 0; n < N; n++){
//       streamIdx = n % STREAM;
//       curr_stream = streams_array[streamIdx];
//
//       // Put function prototype here for guidiance
//       //unroll_Kernel (int C, int H, int W, int K, float* X, float X_unroll)
//       unroll_Kernel<<<gridDim_unroll, MAX_THREAD, 0, curr_stream>>>(C, H, W, K, x.dptr_+ C*H*W*n, x_unroll+streamIdx*H_out*W_out*numWCols);
//
//       // Put function prototype here for guidiance
//       //(float * A, float * B, float * C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
//       matrixMultiply<<<gridDim_mul, blockDim_mul, 2 * TILE_WIDTH * TILE_WIDTH, curr_stream>>>(w.dptr_, x_unroll+(streamIdx * H_out * W_out * numWCols), y.dptr_+ (M * H_out * W_out * n), numWRows, numWCols, numXRows, numXCols, numYRows, numYCols);
//
//
//
//     }
//     // Try to use the cuBlas API, FAILED
//     // const float alpha = 1;
//     // const float beta = 0;
//     // cublasHandle_t handle;
//     // cublasCreate(&handle);
//     // cublasOperation_t transA = CUBLAS_OP_N;
//     // cublasOperation_t transB = CUBLAS_OP_N;
//     //
//     // cublasSgemm(handle, transA, transB, numWRows, numXCols, numWCols, 0, w.dptr_, numWRows, x_unroll, H_out*W_out*numWCols, 0, y.dptr_, numYCols);
//
//     //if(M == 6) cudaProfilerStop();
//
//     cudaFree(x_unroll);
//     MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
//
//
//
//
//     // dim3 gridDim(N, M, Z);
//     // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//     //
//     // cudaMemcpyToSymbol (W, w, M*C*K*K*sizeof(float));
//     //
//     // if (M == 6) cudaProfilerStart();
//     // // Call the kernel
//     // // forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_,w.dptr_, N,M,C,H,W,K);
//     //
//     // // Modified version for running with the constant memory
//     // forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_, N,M,C,H,W,K);
//     //
//     // // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
//     // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
//     //
//     // if(M == 6) cudaProfilerStop();
//
// }
//
// /*
//     This tells mxnet how to do an op when it's not a float.
//     This is not used in the ECE408 project
// */
// template <typename gpu, typename DType>
// void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
// {
//     CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
// }
// }
// }
//
// #endif

///////////////////////////////////////////////////////////////
//  Shared Memory Convolution                               //
//////////////////////////////////////////////////////////////

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include </usr/local/cuda-8.0/targets/x86_64-linux/include/cuda_profiler_api.h>

// Set the tile width, use 16 for now since it is suggested in the textbook
#define TILE_WIDTH 16

// TEST: Try to declare the constant memory stored array for kernel
#define MAX_KERNEL_SIZE 2400
__constant__ float MASK[MAX_KERNEL_SIZE];


namespace mxnet
{
namespace op
{

// The kernel code
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

// Modified version for running with the constant memory
// __global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
// {

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

    h_base = floor(1.0*blockIdx.z / W_grid) * TILE_WIDTH; // block starting index
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH;

    h = h_base + h0; // actual starting index
    w = w_base + w0;

    float acc = 0.;

    int x_input_size = TILE_WIDTH + K - 1;

    // Shared memory implememntation with both input X and Weight matrix
    extern __shared__ float sharemem[];

    // float* share_X = &sharemem[0]; // startting at zero
    // float* share_Mask = &sharemem[x_input_size*x_input_size]; // starting after x input matrix

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // Modified version for running with the constant memory
    #define k4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]



      for (c = 0; c < C; c++) {

        // load input X into shared memory
        for(int i = h; i < h_base + x_input_size; i += TILE_WIDTH){
          for(int j = w; j < w_base + x_input_size; j += TILE_WIDTH){
            if((i<H) && (j<W) && (n<B) && (m<M)){
              sharemem[(i - h_base) * x_input_size + j - w_base] = x4d(n,c,i,j);
            }else{
              sharemem[(i - h_base) * x_input_size + j - w_base] = 0.0f;
            }
          }
        }
        __syncthreads();
        if ( (h < H_out) && (w < W_out)) {

        for (p = 0; p < K; p++){
          for (q = 0; q < K; q++){
          //  if (h0+p < x_input_size && w0+q < x_input_size){
              acc += sharemem[(h0+p)*x_input_size + (w0 + q)]*k4d(m,c,p,q);
              //acc += sharemem[(h0+p)*x_input_size + (w0 + q)]*share_Mask[p*K + q];
          //  }
          }
        }
       }
        __syncthreads();

      // y4d(n, m, h, w) = acc;
      if ( (h < H_out) && (w < W_out)) {
         y4d(n, m, h, w) = acc;
      }
      // }

    #undef y4d
    #undef x4d
    #undef k4d
}
}

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

    cudaMemcpyToSymbol (MASK, w.dptr_, M*C*K*K*sizeof(float));

    if (M == 6) cudaProfilerStart();
    // Call the kernel
    int x_input_size = TILE_WIDTH + K - 1;
    int sharemem_size = (x_input_size*x_input_size) * sizeof(float);
    //int sharemem_size = (x_input_size*x_input_size+K*K) * sizeof(float);
    forward_kernel<<<gridDim, blockDim, sharemem_size>>>(y.dptr_,x.dptr_,w.dptr_, N,M,C,H,W,K);

    // Modified version for running with the constant memory
    //forward_kernel<<<gridDim, blockDim, 0>>>(y.dptr_,x.dptr_, N,M,C,H,W,K);

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

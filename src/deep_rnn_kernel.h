/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */

#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_

#include <fstream>
#include <thrust/transform_reduce.h>
#include <cassert>

#include "utility_cu.h"

#define CUDA_8_0
#define NORM_THREADS 256



////////////////////////////////////// atomicAdd (double) //////////////////////////////////////
////////////////////////////////////// atomicAdd (double) //////////////////////////////////////
////////////////////////////////////// atomicAdd (double) //////////////////////////////////////
//atomic add for doubles,since undefined in cuda
__device__ double atomicAddDouble(double* address, double val);
__device__ double atomicAdd(double* address, double val);



////////////////////////////////////// Input data formatting kernels //////////////////////////////////////
////////////////////////////////////// Input data formatting kernels //////////////////////////////////////
////////////////////////////////////// Input data formatting kernels //////////////////////////////////////

__global__ void VocabTo01Kernel(int *p_device_vocab_indices_01, int *p_device_vocab_indices, int total_length);
__global__ void VocabToNonMinus1Kernel(int *p_device_vocab_indices_non_minus1, int *p_device_vocab_indices, int total_length);


////////////////////////////////////// Forward Prop kernels //////////////////////////////////////
////////////////////////////////////// Forward Prop kernels //////////////////////////////////////
////////////////////////////////////// Forward Prop kernels //////////////////////////////////////

/* Forward sigmoid kernel */
__global__ void ForwardSigmoidKernel(float *p_device_final, float *p_tmp1, float *p_tmp2, float *p_device_bias, int hiddenstate_size);
__global__ void ForwardSigmoidKernel(double *p_device_final, double *p_tmp1, double *p_tmp2, double *p_device_bias, int hiddenstate_size);


template<typename T>
__global__ void ForwardSigmoidKernelSmall(T *p_device_final, T *p_tmp_1, T *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    double temp_val = p_tmp_1[index] + p_device_bias[idx];
    p_device_final[index] = 1.0 / (1.0 + exp(-1.0 * temp_val));
  }
}




template <typename T>
__global__ void ForwardSigmoidKernelFeed(T *p_device_final, T *p_tmp1, T *p_tmp2, T *p_tmp3, T *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    T tmp_value = p_tmp1[index] + p_tmp2[index] + p_tmp3[index] + p_device_bias[idx];
    p_device_final[index] = 1.0 / (1.0 + exp(-1.0 * tmp_value));
  }
}


/* Forward tanh kernel */
__global__ void ForwardTanhKernel(float *p_device_final, float *p_tmp1, float *p_tmp2, float *p_device_bias, int hiddenstate_size);
__global__ void ForwardTanhKernel(double *p_device_final, double *p_tmp1, double *p_tmp2, double *p_device_bias, int hiddenstate_size);


template <typename T>
__global__ void ForwardTanhKernelFeed(T *p_device_final, T *p_tmp1, T *p_tmp2, T *p_tmp3, T *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    T tmp_value = p_tmp1[index] + p_tmp2[index] + p_tmp3[index] + p_device_bias[idx];
    p_device_final[index] = ComputeTanh(tmp_value);
  }
  return;
}


/* Forward c_t kernel */
template <typename T>
__global__ void ForwardCTKernel(T *p_device_c_t, T *p_device_f_t, T *p_device_c_t_prev, T *p_device_i_t, T *p_device_c_prime_t_tanh, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_c_t[index] = p_device_f_t[index] * p_device_c_t_prev[index] + p_device_i_t[index] * p_device_c_prime_t_tanh[index];
  }
}


template <typename T>
__global__ void ForwardCTKernelTree(T *p_device_c_t, T *p_device_f_t_1, T *p_device_c_t_prev_1, T *p_device_f_t_2, T *p_device_c_t_prev_2, T *p_device_i_t, T *p_device_c_prime_t_tanh, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_c_t[index] = p_device_f_t_1[index] * p_device_c_t_prev_1[index] + p_device_f_t_2[index] * p_device_c_t_prev_2[index] + p_device_i_t[index] * p_device_c_prime_t_tanh[index];
  }
}



/* Forward h_t kernel */
__global__ void ForwardHTKernel(float *p_device_h_t, float *p_device_o_t, float *p_device_c_t, int hiddenstate_size);
__global__ void ForwardHTKernel(double *p_device_h_t, double *p_device_o_t, double *p_device_c_t, int hiddenstate_size);





/* Zero c_t and h_t */
template <typename T>
__global__ void ZeroCTAndHT(T *p_device_h_t, T *p_device_c_t, int *p_device_vocab_indices_01, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_h_t[index] = p_device_h_t[index] * p_device_vocab_indices_01[blockIdx.x];
    p_device_c_t[index] = p_device_c_t[index] * p_device_vocab_indices_01[blockIdx.x];
  }
}










// CHECK: OK //
// softmax kernel to preprocess data
template <typename T>
__global__ void VocabSoftmaxKernel(int *p_device_vocab_indices, int *p_device_vocab_indices_01, T *p_device_vocab_indices_01_float, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    if (-1 == p_device_vocab_indices[i]) {
      p_device_vocab_indices[i] = 0;
      p_device_vocab_indices_01[i] = 0;
      p_device_vocab_indices_01_float[i] = 0;
    } else {
      p_device_vocab_indices_01[i] = 1;
      p_device_vocab_indices_01_float[i] = 1;
    }
  }
}




////////////////////////////////////// Template kernels //////////////////////////////////////
////////////////////////////////////// Template kernels //////////////////////////////////////
////////////////////////////////////// Template kernels //////////////////////////////////////

// Kernel for zeroing the w gradient
// length the special length for w gradient
// CHECK: OK //
template <typename T>
__global__ void ZeroWGradientKernel(T *p_device_w_grad, int *p_device_vocab_indices_m1, int hiddenstate_size, int total_length) {
  for (int j = blockIdx.y; j < total_length; j += gridDim.y) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hiddenstate_size) {
      p_device_w_grad[IDX2C(idx, p_device_vocab_indices_m1[j], hiddenstate_size)] = 0;
    }
  }
}


////////////////////////////////////// Backprop kernels //////////////////////////////////////
////////////////////////////////////// Backprop kernels //////////////////////////////////////
////////////////////////////////////// Backprop kernels //////////////////////////////////////

// DErrtCTKernel
__global__ void DErrtCTKernel(float *p_device_d_errt_ct, float *p_device_d_errn_to_t_ht, float *p_device_o_t, float *p_device_c_t, int hiddenstate_size);
__global__ void DErrtCTKernel(double *p_device_d_errt_ct, double *p_device_d_errn_to_t_ht, double *p_device_o_t, double *p_device_c_t, int hiddenstate_size);


/* DeviceErrnToTOTKernel */
__global__ void DErrnToTOTKernel(float *p_device_d_errn_to_t_ot, float *p_device_d_errn_to_t_ht, float *p_device_o_t, float *p_device_c_t, int hiddenstate_size);
__global__ void DErrnToTOTKernel(double *p_device_d_errn_to_t_ot, double *p_device_d_errn_to_t_ht, double *p_device_o_t, double *p_device_c_t, int hiddenstate_size);


// DErrnToTFTITKernel
template <typename T>
__global__ void DErrnToTFTITKernel(T *p_device_d_errn_to_t, T *p_device_d_errn_to_t_ct, T *p_device_single_err, T *p_device_double_err, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_d_errn_to_t[index] = p_device_d_errn_to_t_ct[index] * p_device_single_err[index] * p_device_double_err[index] * (1 - p_device_double_err[index]);
  }
}


// DErrnToTTanhcptKernel
template <typename T>
__global__ void DErrnToTTanhcptKernel(T *p_device_d_errn_to_t_tanhcpt, T *p_device_d_errn_to_t_ct, T *p_device_i_t, T *p_device_c_prime_t_tanh, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_d_errn_to_t_tanhcpt[index] = p_device_d_errn_to_t_ct[index] * p_device_i_t[index] * (1 - p_device_c_prime_t_tanh[index] * p_device_c_prime_t_tanh[index]);
  }
}



// ZeroColumnsKernel
template <typename T>
__global__ void ZeroColumnsKernel(int hiddenstate_size, T *p_device_mat, int *p_device_vec, T *p_device_mat_final) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    p_device_mat_final[IDX2C(idx, blockIdx.x, hiddenstate_size)] = p_device_mat[IDX2C(idx, blockIdx.x, hiddenstate_size)] * \
                                                                   p_device_vec[blockIdx.x];
  }
}

// AddFourMatricsKernel
template <typename T>
__global__ void AddFourMatricesKernel(T *p_device_final, T *p_device_mat1, T *p_device_mat2, T *p_device_mat3, T *p_device_mat4, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_final[index] = p_device_mat1[index] + p_device_mat2[index] + p_device_mat3[index] + p_device_mat4[index];
  }    
}


// ElementwiseMultKernel
template <typename T>
__global__ void ElementwiseMultKernel(T *p_device_mat1, T *p_device_mat2, T *p_device_final, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    p_device_final[IDX2C(idx, blockIdx.x, hiddenstate_size)] = p_device_mat1[IDX2C(idx, blockIdx.x, hiddenstate_size)] * p_device_mat2[IDX2C(idx, blockIdx.x, hiddenstate_size)];
  }
}




template <typename T>
__global__ void SparseLookupKernel(T *p_device_lookup, T *p_device_w, int *p_device_vocab_indices, int minibatch_size, int hiddenstate_size) {
  int index = threadIdx.x + blockIdx.y * blockDim.x;
  if (index < hiddenstate_size) {
    // p_device_vocab_indices[blockIdx.x], as minibatch = 64, blockIdx.x = 0...63, 
    p_device_lookup[IDX2C(index, blockIdx.x, hiddenstate_size)] = p_device_w[IDX2C(index, p_device_vocab_indices[blockIdx.x], hiddenstate_size)];
  }
}



/* WGradientKernel */
__global__ void WGradientKernel(float *p_device_w_grad, int *p_device_vocab_indices, float *p_tmp1, float *p_tmp2, float *p_tmp3, float *p_tmp4, int hiddenstate_size);
__global__ void WGradientKernel(double *p_device_w_grad, int *p_device_vocab_indices, double *p_tmp1, double *p_tmp2, double *p_tmp3, double *p_tmp4, int hiddenstate_size);


/* WGradientKernelDropout */
__global__ void WGradientKernelDropout(float *p_device_w_grad, int *p_device_vocab_indices, float *p_tmp1, float *p_tmp2, float *p_tmp3, float *p_tmp4, int hiddenstate_size, float *p_device_dropout_mask, float rate);
__global__ void WGradientKernelDropout(double *p_device_w_grad, int *p_device_vocab_indices, double *p_tmp1, double *p_tmp2, double *p_tmp3, double *p_tmp4, int hiddenstate_size, double *p_device_dropout_mask, double rate);


template<typename T>
__global__ void WSmallGradientKernel(T *p_device_small_w_grad, int *p_device_reverse_unique_indices, T *p_tmp1, T *p_tmp2, T *p_tmp3, T *p_tmp4, int *p_device_vocab_indices, int lstm_size, int minibatch_size) {

  // start at the thread index
  int i_start = threadIdx.x;
  // end at dim
  int i_end = lstm_size;
  // the block dimension (aka the number of threads in the block) is the step
  int i_step = blockDim.x;

  for (int k = blockIdx.x; k < minibatch_size; k += gridDim.x) {
    int vocab_index = p_device_vocab_indices[k];
    for (int i = i_start; i < i_end; i += i_step) {
      T sum = p_tmp1[IDX2C(i, k, lstm_size)] + p_tmp2[IDX2C(i, k, lstm_size)] + p_tmp3[IDX2C(i, k, lstm_size)] + p_tmp4[IDX2C(i, k, lstm_size)];
      atomicAdd(&(p_device_small_w_grad[IDX2C(i, p_device_reverse_unique_indices[vocab_index], lstm_size)]), sum);
    }
  }
}


template <typename T>
__global__ void WSmallGradientKernelDropout(T *p_device_small_w_grad, int *p_device_reverse_unique_indices, T *p_tmp1, T *p_tmp2, T *p_tmp3, T *p_tmp4, int *p_device_vocab_indices, int lstm_size, int minibatch_size, T *p_device_dropout_mask, T rate) {
  int i_start = threadIdx.x;
  int i_end = lstm_size;
  int i_step = blockDim.x;

  for (int k = blockIdx.x; k < minibatch_size; k += gridDim.x) {
    int vocab_index = p_device_vocab_indices[k];
    for (int i = i_start; i < i_end; i += i_step) {
      T sum = p_tmp1[IDX2C(i, k, lstm_size)] + p_tmp2[IDX2C(i, k, lstm_size)] + p_tmp3[IDX2C(i, k, lstm_size)] + p_tmp4[IDX2C(i, k, lstm_size)];
      sum = sum * (rate > p_device_dropout_mask[IDX2C(i, k, lstm_size)]) * (1 / rate);
      atomicAdd(&(p_device_small_w_grad[IDX2C(i, p_device_reverse_unique_indices[vocab_index], lstm_size)]), sum);
    }
  }
}


////////////////////////////////////// Softmax kernels //////////////////////////////////////
////////////////////////////////////// Softmax kernels //////////////////////////////////////
////////////////////////////////////// Softmax kernels //////////////////////////////////////

#define SOFTMAX_THREADS 256

template <typename T>
__device__ void WarpReduceSum(volatile T* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}


template <typename T>
__device__ void WarpReduceMax(volatile T* sdata, int tid) {
  sdata[tid] = (sdata[tid] > sdata[32 + tid]) ? sdata[tid] : sdata[32 + tid];
  sdata[tid] = (sdata[tid] > sdata[16 + tid]) ? sdata[tid] : sdata[16 + tid];
  sdata[tid] = (sdata[tid] > sdata[8 + tid]) ? sdata[tid] : sdata[8 + tid];
  sdata[tid] = (sdata[tid] > sdata[4 + tid]) ? sdata[tid] : sdata[4 + tid];
  sdata[tid] = (sdata[tid] > sdata[2 + tid]) ? sdata[tid] : sdata[2 + tid];
  sdata[tid] = (sdata[tid] > sdata[1 + tid]) ? sdata[tid] : sdata[1 + tid];
  return;
}


template <typename T>
__global__ void TrainPerplexityKernel(int *p_device_output_vocab_indices_single, int *p_device_output_vocab_indices_01_single, T *p_device_outputdist, double *train_perplexity, int minibatch_size, int output_vocab_size) {
  for (int i = 0; i < minibatch_size; ++i) {
    if (1 == p_device_output_vocab_indices_01_single[i]) {
      train_perplexity[0] += log((double)p_device_outputdist[IDX2C(p_device_output_vocab_indices_single[i], i, output_vocab_size)]);
    }
  }
}

/*
  Each thread in a block gets a location in the buffer. Initially the max element is stored in this location
  For buffer one extra slot is allocated to store the true max of the buffer
  Each block does one output dist column, so for a minibatch of 128, simple call this with dim = 20000 and blocks = 128
  column major storage is necessary for this
  adapted from torch
  this does summing and exping all in one go, so no thrust or column of 1's needed
*/
template <typename T>
__global__ void OutputdistOverflowPreventionKernel(T *p_output, T *p_input, int dim) {
  // shared memory for the block, this must be the number of threads per block in size
  __shared__ T buffer[SOFTMAX_THREADS];
  // get the block index
  int k = blockIdx.x;
  // all threads in block start from same index
  T *p_input_k = p_input + k * dim;
  // all threads in block start from same index
  T *p_output_k = p_output + k * dim;

  // start at the thread index
  int i_start = threadIdx.x;
  // end at dim
  int i_end = dim;
  // the block dimension (aka the number of threads in the block) is the step
  int i_step = blockDim.x;
  const int tid = threadIdx.x;

  // get the max element for each thread's assigned locations and put them in the buffer
  // dim elements are covered in this reduction
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i = i_start; i < i_end; i += i_step) {
    T z = p_input_k[i];
    if (buffer[threadIdx.x] < z) {
      buffer[threadIdx.x] = z;
    }
  }

  __syncthreads();

  // reduce
  // first thread goes through and finds the max element in the buffer
  // after this stage the max element for dim items is found
  for (int stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
    }
    __syncthreads();
  }

  __syncthreads();

  // sum
  // now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
  T max_k = buffer[0];
  // this must be here
  __syncthreads();
  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    // subtract the max from the input, then exp it for the softmax
    T z = ComputeExp(p_input_k[i] - max_k);
    // keep a running sum of these values for the normalization constant
    buffer[threadIdx.x] += z;
    // set the output as this value, then get ready to divide by the sum again
    p_output_k[i] = z;
  }

  __syncthreads();

  // reduce
  // now sum all the elements in the buffer, for the normalization constant
  for (int stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }

    __syncthreads();
  }

  __syncthreads();

  // normalize the softmax
  T sum_k = buffer[0];
  for (int i = i_start; i < i_end; i += i_step) {
    p_output_k[i] = p_output_k[i] / sum_k;
  }
}


template <typename T, typename T2>
__global__ void OutputdistPerplexityKernel(T2 *p_output, T *p_input, int dim, bool print_partition_function, double *p_device_partition_vals) {
  // shared memory for the block, this must be the number of threads per block in size
  __shared__ double buffer[SOFTMAX_THREADS];

  // get the block index
  int k = blockIdx.x;

  // all threads in block start from same index
  T *p_input_k = p_input + k * dim;
  // again all threads in block start from same index
  T2 *p_output_k = p_output + k * dim;

  // start at the thread index
  int i_start = threadIdx.x;
  // end at dim
  int i_end = dim;
  // the block dimension (aka the number of threads in the block) is the step
  int i_step = blockDim.x;
  const int tid = threadIdx.x;

  // get the max element for each thread's assigned locations and put them in the buffer
  // dim elements are covered in this reduction
  buffer[threadIdx.x] = -DBL_MAX;
  for (int i = i_start; i < i_end; i += i_step) {
    double z = p_input_k[i];
    if (buffer[threadIdx.x] < z) {
      buffer[threadIdx.x] = z;
    }
  }

  __syncthreads();

  // reduce
  // first thread goes through and finds the max element in the buffer
  // after this stage the max element for dim items is found
  for (int stride = SOFTMAX_THREADS / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
    }
    __syncthreads();
  }

  if (tid < 32) {
    WarpReduceMax(buffer, tid);
  }

  __syncthreads();

  // sum, now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant
  double max_k = buffer[0];
  __syncthreads();
  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    double z = ComputeExp(p_input_k[i] - max_k);
    // keep a running sum of these values for the normalization constant
    buffer[threadIdx.x] += z;
    // set the output as this value, then get ready to divide by the sum again
    p_output_k[i] = z;
  }

  __syncthreads();

  // reduce
  // now sum all the elements in the buffer, for the normalization constant
  for (int stride = SOFTMAX_THREADS / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }
    __syncthreads();
  }

  if (tid < 32) {
    WarpReduceSum(buffer, tid);
  }

  __syncthreads();

  // normalize the softmax
  double sum_k = buffer[0];
  for (int i = i_start; i < i_end; i += i_step) {
    p_output_k[i] = ComputeLog(p_output_k[i]) - ComputeLog(sum_k);
  }

  if (print_partition_function && 0 == threadIdx.x) {
    p_device_partition_vals[blockIdx.x] = sum_k;
  }
}




template <typename T>
__global__ void MatrixBiasKernel(int hiddenstate_size, T *p_device_mat, T *p_device_vec, T *p_device_mat_final) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    p_device_mat_final[IDX2C(idx, blockIdx.x, hiddenstate_size)] = p_device_mat[IDX2C(idx, blockIdx.x, hiddenstate_size)] + p_device_vec[idx];
  }
}

struct ExpFunctorGpu {
  __host__ __device__ void operator() (float &x) {
    x = expf(x);  
  }

  __host__ __device__ void operator() (double &x) {
    x = exp(x);
  }
};

/* inverse each element in matrix */
struct InvFunctorGpu {
  template <typename T>
  __host__ __device__ void operator() (T &x) {
    x = 1 / x;
  }
};




template <typename T>
__global__ void ZeroColumnsKernel128(int hiddenstate_size, T *p_device_mat, int *p_device_vec, T *p_device_mat_final) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;  // idx: 0 -> (lstm_size - 1)
  if (idx < hiddenstate_size) {
    p_device_mat_final[IDX2C(idx, blockIdx.x, hiddenstate_size)] = p_device_mat[IDX2C(idx, blockIdx.x, hiddenstate_size)] * p_device_vec[blockIdx.x];
  }
}



// This kernel adds a matrices rows to a matrices columns, which ones depend on the index
// hiddenstate_size refers to the number of rows in p_device_mat_final and also p_device_mat_col
template <typename T>
__global__ void MatrixRowToMatrixColumnKernel(T *p_device_mat_final, T *p_device_mat_col, T *p_device_mat_row, int *p_device_indices, int hiddenstate_size, int output_vocab_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;         // 0 - (output_vocab_size - 1)
  if (idx < hiddenstate_size) {
      p_device_mat_final[IDX2C(idx, blockIdx.x, hiddenstate_size)] = p_device_mat_col[IDX2C(idx, blockIdx.x, hiddenstate_size)] + \
                                                                     p_device_mat_row[IDX2C(p_device_indices[blockIdx.x], idx, output_vocab_size)];
  }
}


/* This kernel adds a matrices columns to a matrices rows, which ones depend on the index */
/* hiddenstate_size refers to the number of rows in p_device_mat_final and also p_device_mat_col */
__global__ void MatrixColumnToMatrixRowKernel(float *p_device_mat_final, float *p_device_mat_col, float *p_device_mat_row, int *p_device_indices, int hiddenstate_size, int output_vocab_size);
__global__ void MatrixColumnToMatrixRowKernel(double *p_device_mat_final, double *p_device_mat_col, double *p_device_mat_row, int *p_device_indices, int hiddenstate_size, int output_vocab_size);


/* add ones to b_d bias unit */
__global__ void AddOnesBDGrad(float *p_device_b_d_grad, int *p_device_output_vocab_indices_01, int *p_device_output_vocab_indices, int minibatch_size);
__global__ void AddOnesBDGrad(double *p_device_b_d_grad, int *p_device_output_vocab_indices_01, int *p_device_output_vocab_indices, int minibatch_size);


////////////////////////////////////// updating parameters //////////////////////////////////////
////////////////////////////////////// updating parameters //////////////////////////////////////
////////////////////////////////////// updating parameters //////////////////////////////////////

struct ScaleFunctor{
  const int minibatch_size_;
  ScaleFunctor(int minibatch_size) : minibatch_size_(minibatch_size) {};

  __host__ __device__ void operator() (float &x) {
    x = (1.0f / minibatch_size_) * x;
  }

  __host__ __device__ void operator() (double &x) {
    x = (1.0 / minibatch_size_) * x;
  }
};


template <typename T>
struct ReScaleNormFunctor{
  const T norm_threshold_;
  const T norm_;

  ReScaleNormFunctor(T norm_threshold, T norm) : norm_threshold_(norm_threshold), norm_(norm) {}

  __host__ __device__ void operator() (T &x) {
    x = (norm_threshold_ / norm_) * x;
  }
};


template <typename T>
__global__ void BasicComputeNormP1(T *p_device_gradient, int size, T *p_result) {
  __shared__ T buffer[NORM_THREADS];
  // start at the thread index
  int i_start = threadIdx.x + blockIdx.x * blockDim.x;
  // end at dim
  int i_end = size;
  // the block dimension (aka the number of threads in the block) is the step
  int i_step = blockDim.x * gridDim.x;
  int tid = threadIdx.x;

  buffer[tid] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    buffer[tid] += (p_device_gradient[i] * p_device_gradient[i]);
  }
  __syncthreads();

  for (int stride = NORM_THREADS / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }
    __syncthreads();
  }

  if (tid < 32) {
    WarpReduceSum(buffer, tid);
  }
  __syncthreads();

  if (0 == tid) {
    p_result[blockIdx.x] = buffer[0];
  }
}


template <typename T>
__global__ void BasicComputeNormP2(T *p_result_tmp, T *p_result_final) {
  __shared__ T buffer[NORM_THREADS];

  int tid = threadIdx.x;
  buffer[tid] = p_result_tmp[tid];
  __syncthreads();

  for (int stride = NORM_THREADS / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }
    __syncthreads();
  }

  if (tid < 32) {
    WarpReduceSum(buffer, tid);
  }
  __syncthreads();

  if (0 == tid) {
    p_result_final[0] = buffer[0];
  }
}


// clip the norm if it is greater than the threshold
template <typename T>
void NormClipGpuV2(thrust::device_ptr<T> &p_thrust_device_gradient, T *p_device_gradient, T norm_threshold, int size, T *p_device_result_tmp, T *p_device_result) {
  T norm;
  BasicComputeNormP1<<<NORM_THREADS, NORM_THREADS>>>(p_device_gradient, size, p_device_result_tmp);
  BasicComputeNormP2<<<1, NORM_THREADS>>>(p_device_result_tmp, p_device_result);
  cudaMemcpy(&norm, p_device_result, 1 * sizeof(T), cudaMemcpyDeviceToHost);
  neural_machine_translation::recent_sum__ = norm;
  // norm = sqrt(sum_i(p_device_gradient[i]^2))
  norm = std::sqrt(norm);

  if (norm > norm_threshold) {
    ReScaleNormFunctor<T> unary_op(norm_threshold, norm);
    // p_thrust_device_gradient[i] = (norm_threshold/norm) * p_thrust_device_gradient[i]
    thrust::for_each(p_thrust_device_gradient, p_thrust_device_gradient + size, unary_op);
  }
}


// for global clipping
template <typename T>
void NormClipGpuV2P1(thrust::device_ptr<T> &p_thrust_device_gradient, T *p_device_gradient, T norm_threshold, int size, T *p_device_result_tmp, T *p_device_result) {
  T norm;
  BasicComputeNormP1<<<NORM_THREADS, NORM_THREADS>>>(p_device_gradient, size, p_device_result_tmp);
  BasicComputeNormP2<<<1, NORM_THREADS>>>(p_device_result_tmp, p_device_result);
  DeviceSyncAll();
  cudaMemcpy(&norm, p_device_result, 1 * sizeof(T), cudaMemcpyDeviceToHost);

  neural_machine_translation::global_norm_clip__ += norm;
  neural_machine_translation::recent_sum__ = norm;
}


/* for global clipping */
template <typename T>
void NormClipGpuV2P2(thrust::device_ptr<T> &p_thrust_device_gradient, T *p_device_gradient, T norm_threshold, int size, T *p_device_result_tmp, T *p_device_result) {
  if (neural_machine_translation::global_norm_clip__ > norm_threshold) {
    ReScaleNormFunctor<T> unary_op(norm_threshold, neural_machine_translation::global_norm_clip__);
    thrust::for_each(p_thrust_device_gradient, p_thrust_device_gradient + size, unary_op);
  }
}


/* kernel for getting scaling the gradient of W by 1 / (minibatch size) */
template <typename T>
__global__ void ScaleWGradient(T *p_device_w_gradient, int *p_device_vocab_indices_m1, int hiddenstate_size, T scale, int total_length) {
  for (int j = blockIdx.y; j < total_length; j += gridDim.y) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hiddenstate_size) {
      const int index = IDX2C(idx, p_device_vocab_indices_m1[j], hiddenstate_size);
      p_device_w_gradient[index] = scale * p_device_w_gradient[index];
    }
  }
  return;
}


template <typename T>
__global__ void IndividualClipWGradient(T *p_device_w_gradient, int *p_device_vocab_indices_m1, int hiddenstate_size, T threshold, int total_length) {
  for (int j = blockIdx.y; j < total_length; j += gridDim.y) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hiddenstate_size) {
      const int index = IDX2C(idx, p_device_vocab_indices_m1[j], hiddenstate_size);
      if (p_device_w_gradient[index] > 0) {
        p_device_w_gradient[index] = (p_device_w_gradient[index] > threshold) ? threshold : p_device_w_gradient[index];
      } else {
        p_device_w_gradient[index] = (p_device_w_gradient[index] < -threshold) ? -threshold : p_device_w_gradient[index];
      }
    }
  }
  return;
}


/* Compute 12 norm of W */
template <typename T>
__global__ void NormWComputeP1(T *p_device_w_gradient, T *p_global_tmpsum, int *p_device_vocab_indices, int hiddenstate_size, int total_length) {
  __shared__ T buffer[NORM_THREADS];

  /* start at the thread index */
  int i_start = threadIdx.x;
  /* end at dim */
  int i_end = hiddenstate_size;
  /* the block dimension (aka the number of threads in the block) is the step */
  int i_step = blockDim.x;
  int tid = threadIdx.x;

  int j_start = blockIdx.x;
  int j_end = total_length;
  int j_step = gridDim.x;
  int bid = blockIdx.x;

  if (0 == tid) {
    p_global_tmpsum[bid] = 0;
  }

  for (int j = j_start; j < j_end; j += j_step) {
    buffer[tid] = 0;
    for (int i = i_start; i < i_end; i += i_step) {
      buffer[tid] += (p_device_w_gradient[IDX2C(i, p_device_vocab_indices[j], hiddenstate_size)] * p_device_w_gradient[IDX2C(i, p_device_vocab_indices[j], hiddenstate_size)]);
    }
    __syncthreads();

    for (int stride = NORM_THREADS / 2; stride > 32; stride >>= 1) {
      if (tid < stride) {
        buffer[tid] += buffer[stride + tid];
      }
      __syncthreads();
    }

    if (tid < 32) {
      WarpReduceSum(buffer, tid);
    }
    __syncthreads();
    if (0 == tid) {
      p_global_tmpsum[bid] += buffer[0];
    }
    __syncthreads();
  }
  return;
}


/* Compute 12 norm of W */
/* NOTE this should be lanched with only 1 block */
template <typename T>
__global__ void NormWComputeP2(T *p_global_tmpsum) {
  __shared__ T buffer[NORM_THREADS];
  int tid = threadIdx.x;

  buffer[tid] = p_global_tmpsum[tid];

  __syncthreads();

  for (int stride = NORM_THREADS / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }
    __syncthreads();
  }

  if (tid < 32) {
    WarpReduceSum(buffer, tid);
  }
  __syncthreads();

  if (0 == tid) {
    p_global_tmpsum[0] = buffer[0];
  }
  return;
}



/* v2 with custom w gradient clipping */
template <typename T>
void NormClipWGpuV2(T *p_device_global_w_sum, T *p_device_grad, int *p_device_vocab_indices_m1, T norm_threshold, int total_length, int hiddenstate_size) {
  T norm;
  NormWComputeP1<<<NORM_THREADS, NORM_THREADS>>>(p_device_grad, p_device_global_w_sum, p_device_vocab_indices_m1, hiddenstate_size, total_length);
  NormWComputeP2<<<1, NORM_THREADS>>>(p_device_global_w_sum);
  cudaMemcpy(&norm, p_device_global_w_sum, 1 * sizeof(T), cudaMemcpyDeviceToHost);
  norm = std::sqrt(norm);

  if (norm > norm_threshold) {
    int threads_per_block = 256;
    int num_block = (hiddenstate_size + threads_per_block - 1) / threads_per_block;
    dim3 kernel(num_block, 256, 1);
    T scalar = (norm_threshold / norm);
    ScaleWGradient<<<kernel, threads_per_block>>>(p_device_grad, p_device_vocab_indices_m1, hiddenstate_size, scalar, total_length);
  }
  return;
}



/* v2 with custom w gradient clipping */
template <typename T>
void NormClipWGpuV2P2(T *p_device_global_w_sum, T *p_device_grad, int *p_device_vocab_indices_m1, T norm_threshold, int total_length, int hiddenstate_size) {
  DeviceSyncAll();
  if (neural_machine_translation::global_norm_clip__ > norm_threshold) {
    int threads_per_block = 256;
    int num_block = (hiddenstate_size + threads_per_block - 1) / threads_per_block;
    dim3 kernel(num_block, 256, 1);
    T scalar = (norm_threshold / neural_machine_translation::global_norm_clip__);
    ScaleWGradient<<<kernel, threads_per_block>>>(p_device_grad, p_device_vocab_indices_m1, hiddenstate_size, scalar, total_length);
  }
  return;
}


/* kernel for updating the w gradient */
template <typename T>
__global__ void UpdateWGradient(T *p_device_w, T *p_device_w_gradient, int *p_device_vocab_indices_m1, T learning_rate, int hiddenstate_size, int total_length) {
  for (int j = blockIdx.y; j < total_length; j += gridDim.y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < hiddenstate_size) {
      int index = IDX2C(idx, p_device_vocab_indices_m1[j], hiddenstate_size);
      p_device_w[index] = learning_rate * p_device_w_gradient[index] + p_device_w[index];
    }
  }
  return;
}


template <typename T>
__global__ void UpdateSparseGradient(T *p_device_mat, T *p_device_small_grad, int *p_device_unique_indices, int current_num_unique, T learning_rate, int lstm_size) {
  int i_start = threadIdx.x;
  int i_end = lstm_size;
  int i_step = blockDim.x;

  for (int k = blockIdx.x; k < current_num_unique; k += gridDim.x) {
    int vocab_index = p_device_unique_indices[k];
    for (int i = i_start; i < i_end; i += i_step) {
      p_device_mat[IDX2C(i, vocab_index, lstm_size)] += learning_rate * p_device_small_grad[IDX2C(i, k, lstm_size)];
    }
  }
}


template <typename T>
__global__ void AddGradVecs(T *p_vec1, T *p_vec2, T learning_rate, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    p_vec1[idx] = learning_rate * p_vec2[idx] + p_vec1[idx];
  }
}

////////////////////////////////////// Decoder Stuff //////////////////////////////////////
////////////////////////////////////// Decoder Stuff //////////////////////////////////////
////////////////////////////////////// Decoder Stuff //////////////////////////////////////

template <typename T>
__global__ void OnesMatKernel(T *mat, int size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    mat[i] = 1;
  }
}






////////////////////////////////////// truncated softmax Stuff //////////////////////////////////////
////////////////////////////////////// truncated softmax Stuff //////////////////////////////////////
////////////////////////////////////// truncated softmax Stuff //////////////////////////////////////

/* called when updating parameters */
template <typename T>
__global__ void TruncDGradNonshort(T *p_device_subset_d_grad, T *p_device_d, int *p_device_vocab_mappings, int hiddenstate_size, int trunc_size, int output_vocab_size, T learning_rate, int shortlist_size) {
  for (int j = blockIdx.x + shortlist_size; j < trunc_size; j += gridDim.x) {
    for (int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
      p_device_d[IDX2C(p_device_vocab_mappings[j - shortlist_size], i, output_vocab_size)] += learning_rate * p_device_subset_d_grad[IDX2C(j, i, trunc_size)];
    }
  }
  return;
}


template <typename T>
__global__ void TruncDGradShort(T *p_device_subset_d_grad, T *p_device_subset_d, int hiddenstate_size, int shortlist_size, T learning_rate, int trunc_size) {
  for (int j = blockIdx.x; j < shortlist_size; j += gridDim.x) {
    for (int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
      p_device_subset_d[IDX2C(j, i, trunc_size)] += learning_rate * p_device_subset_d_grad[IDX2C(j, i, trunc_size)];
    }  
  }
  return;
}


/* called when finished training before parameters are written to a file */
template <typename T>
__global__ void LoadShortlistD(T *p_device_subset_d, T *p_device_d, int hiddenstate_size, int trunc_size, int output_vocab_size, int shortlist_size) {
  for (int j = blockIdx.x; j < shortlist_size; j += gridDim.x) {
    for (int i = threadIdx.x; i < hiddenstate_size; i += blockDim.x) {
      p_device_d[IDX2C(j, i, output_vocab_size)] = p_device_subset_d[IDX2C(j, i, trunc_size)];
    }
  }
  return;
}



/* scales before normalization stage */
/* call in place of overflow kernel */
template <typename T>
__global__ void OutputdistTruncatedKernel(T *p_output, T *p_input, int dim, T sample_rate, int shortlist_size_plus) {
  /* shared memory for the block, this must be the number of threads per block in size */
  __shared__ T buffer[SOFTMAX_THREADS];
  /* get the block index */
  int k = blockIdx.x;
  /* all threads in block start from same index */
  T *p_input_k = p_input + k * dim;
  /* all threads in block start from same index */
  T *p_output_k = p_output + k * dim;

  /* start at the thread index */
  int i_start = threadIdx.x;
  /* end at dim */
  int i_end = dim;
  /* the block dimension (aka the number of threads in the block) is the step */
  int i_step = blockDim.x;
  const int tid = threadIdx.x;

  /* get the max element for each thread's assigned locations and put them in the buffer */
  /* dim elements are covered in this reduction */
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i = i_start; i < i_end; i += i_step) {
    T z = p_input_k[i];
    if (buffer[threadIdx.x] < z) {
      buffer[threadIdx.x] = z;
    }
  }

  __syncthreads();

  /* reduce */
  /* first thread goes through and finds the max element in the buffer */
  /* after this stage the max element for dim items is found */
  for (int stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] = (buffer[tid] > buffer[stride + tid]) ? buffer[tid] : buffer[stride + tid];
    }

    __syncthreads();
  }

  __syncthreads();

  /* sum */
  /* now go through all the dim elements and subtract the max from the element, keep a running sum for the normalization constant */
  T max_k = buffer[0];
  /* this must be here */
  __syncthreads();
  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    T z;
    if (i >= shortlist_size_plus) {
      z = sample_rate * ComputeExp(p_input_k[i] - max_k);
    } else {
      z = ComputeExp(p_input_k[i] - max_k);
    }

    /* keep a running sum of these values for the normalization constant */
    buffer[threadIdx.x] += z;
    /* set the output as this value, then get ready to divide by the sum again */
    p_output_k[i] = z;
  }

  __syncthreads();

  /* reduce */
  /* now sum all the elements in the buffer, for the normalization constant */
  for (int stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }
    __syncthreads();
  }

  __syncthreads();

  /* normalize the softmax */
  T sum_k = buffer[0];
  for (int i = i_start; i < i_end; i += i_step) {
    p_output_k[i] = p_output_k[i] / sum_k;
  }
  return;
}






////////////////////////////////////// Dropout Stuff //////////////////////////////////////
////////////////////////////////////// Dropout Stuff //////////////////////////////////////
////////////////////////////////////// Dropout Stuff //////////////////////////////////////

// for forward and backward pass for error and h_t in LSTM
template <typename T>
__global__ void DropoutKernel(T *p_device_dropout_mask, T rate, T *p_device_final, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    p_device_final[i] = (p_device_dropout_mask[i] < rate) * (1/rate) * p_device_final[i];
  }
}



////////////////////////////////////// Attention Model //////////////////////////////////////
////////////////////////////////////// Attention Model //////////////////////////////////////
////////////////////////////////////// Attention Model //////////////////////////////////////
__global__ void TanhKernel(float *p_device_in, float *p_device_out, int total_length);
__global__ void TanhKernel(double *p_device_in, double *p_device_out, int total_length);


// CHECK: OK //
template <typename T>
__global__ void SigmoidKernel(T *p_device_in, T *p_device_out, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    p_device_out[i] = 1.0 / (1.0 + ComputeExp(-1.0 * p_device_in[i]));  
  }
}


// CHECK: OK //
// Batch information is in the form: [sentent lengths][offsets]
template <typename T>
__global__ void AlignmentPosKernel(T *p_device_in, T *p_device_out, int total_length, int *p_device_batch_information) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    p_device_out[i] = p_device_batch_information[i] * p_device_in[i];  
  }
}


// CHECK: OK //
template <typename T>
__global__ void LowerUpperKernel(T *p_device_p_t, int *p_device_lower_upper, int d, int *p_device_batch_information, int minibatch_size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < minibatch_size; i += gridDim.x * blockDim.x) {
    p_device_lower_upper[IDX2C(0,i,2)] = (0 > (int)(p_device_p_t[i]) - d) ? 0 : ((int)(p_device_p_t[i]) - d);
    p_device_lower_upper[IDX2C(1,i,2)] = ((p_device_batch_information[i] - 1) < (int)(p_device_p_t[i]) + d) ? (p_device_batch_information[i] - 1) : ((int)(p_device_p_t[i]) + d);
  }
}


// For getting viterbi alignments
template <typename T>
__global__ void GetViterbiAlignmentKernel(T *p_device_alignments, int *p_device_indices, int d, int minibatch_size, int *p_device_final_results) {
  int minibatch_index = threadIdx.x;
  T max_val = -1;
  int max_index = -1;
  for (int i = 0; i < 2 * d + 1; ++i) {
    if (max_val < p_device_alignments[IDX2C(minibatch_index, i, minibatch_size)]){
      max_val = p_device_alignments[IDX2C(minibatch_index, i, minibatch_size)];
      max_index = p_device_indices[IDX2C(minibatch_index, i, minibatch_size)];
    }
  }

  p_device_final_results[minibatch_index] = max_index;
}


/* create indices */
__global__ void CreateIndicesKernel(int *p_device_indices, int d, int minibatch_size, int *p_device_lower_upper, int *p_device_01_mask);


//
__global__ void SetupReverseIndices(int *p_device_reverse_unique_indices, int *p_device_unique_indices, int current_num_unique);


/* 
  p_device_total_hs_mat is the length of the source length, where each pointer points to h_s minibatch at that source index

  parallelism works as follows:
  each block copies one h_s vector for each minibatch

  p_device_indices is the size of (2 * d + 1) * minibatch of ints
  -1 index means that the alignment is not pointing to a valid source index, will need to zero this out in the exped scores

  change the parallelism to make each block do 2 * d + 1 operation??? Benchmark this
*/
// CHECK: OK //

template <typename T>
__global__ void LoadInHSKernel(T **p_device_total_hs_mat, int d, T *p_device_hs_mat, int *p_device_indices, int minibatch_size, int lstm_size, int *p_device_batch_information) {
  // each block is responsible for copying one h_s vector into the current h_s
  for (int i = blockIdx.x; i < (2 * d + 1) * minibatch_size; i += gridDim.x) {
    int minibatch_index = i % minibatch_size;
    int source_index = p_device_indices[i];

    if (-1 != source_index) {
      for (int j = threadIdx.x; j < lstm_size; j += blockDim.x) {
        //p_device_hs_mat[IDX2C(j, i, lstm_size)] = p_device_total_hs_mat[source_index + p_device_batch_information[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, lstm_size)];
        // i.e. source_sent = (a b c d e), then source_sent should be modified to (0 0 0 0 e d c b a)
        // p_device_batch_information[minibatch_index] = 5
        // source_index = 1 (we want to get *b*)
        // p_device_batch_information[minibatch_size + minibatch_index] = 4
        // so (5 - 1 - 1 + 4) = 7 
        p_device_hs_mat[IDX2C(j, i, lstm_size)] = p_device_total_hs_mat[p_device_batch_information[minibatch_index] - 1 - source_index + p_device_batch_information[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, lstm_size)];
      }
    } else {
      for (int j = threadIdx.x; j < lstm_size; j += blockDim.x) {
        p_device_hs_mat[IDX2C(j, i, lstm_size)] = 0;  
      }
    }
  }
}

/*
template<typename dType>
__global__
void LoadInHSKernel(dType **d_total_hs_mat, int D, dType *d_hs_mat, int *d_indices, int minibatch_size, int LSTM_size, int *d_batch_info) {

    //each block is responsible for copying one h_s vector into the current h_s
    for (int i = blockIdx.x; i < (2 * D + 1)*minibatch_size; i += gridDim.x) {
        int minibatch_index = i % minibatch_size;
        int source_index = d_indices[i];
        if (source_index != -1) {
            for (int j = threadIdx.x; j < LSTM_size; j += blockDim.x) {
                d_hs_mat[IDX2C(j, i, LSTM_size)] = d_total_hs_mat[source_index + d_batch_info[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, LSTM_size)];
            }
        }
        else {
            for (int j = threadIdx.x; j < LSTM_size; j += blockDim.x) {
                d_hs_mat[IDX2C(j, i, LSTM_size)] = 0;
            }
        }
    }
} */



/*
  alignment are stored in the following way:
  [minibatch, minibatch, minibatch, ...]

  each thread does a reduction for a minibatch
*/
// CHECK: OK //
template <typename T>
__global__ void AlignmentReductionKernel(T *p_device_alignments, int lstm_size, int minibatch_size, int d, T sigma_sq, T *p_device_p_t, int *p_device_indices, T *p_device_cached_exp) {

  int minibatch_index = threadIdx.x;
  if (minibatch_index < minibatch_size) {
    T sum = 0;
    T max_value = 0;
    // find max_value in p_device_alignments
    for (int i = 0; i < 2 * d + 1; ++i) {
      if (-1 != p_device_indices[minibatch_index + minibatch_size * i]) {
        if (p_device_alignments[minibatch_index + minibatch_size * i] > max_value) {
          max_value = p_device_alignments[minibatch_index + minibatch_size * i];
        }
      }
    }

    // calculate exp(score(h_t,h_s))
    // calculate sum_s^'(exp(score(h_t,h_s^')))
    // score(h_t,h_s) = score(h_t,h_s) - max_value
    for (int i = 0; i < 2 * d + 1; ++i) {
      if (-1 != p_device_indices[minibatch_index + minibatch_size * i]) {
        p_device_alignments[minibatch_index + minibatch_size * i] = exp(p_device_alignments[minibatch_index + minibatch_size * i] - max_value);
        sum += p_device_alignments[minibatch_index + minibatch_size * i];
      } else {
        p_device_alignments[minibatch_index + minibatch_size * i] = 0;
      }
    }

    for (int i = 0; i < 2 * d + 1; ++i) {
      if (-1 != p_device_indices[minibatch_index + minibatch_size * i]) {
        // tmp = exp(-(s - p_t)^2/ 2*sigma_sq^2), tmp in (0,1)
        T tmp = exp((-1 * ComputePow((p_device_p_t[minibatch_index] - p_device_indices[minibatch_index + minibatch_size * i]), (T)2.0)) / (2 * sigma_sq));

        if (0 != sum) {
          p_device_alignments[minibatch_index + minibatch_size * i] = (p_device_alignments[minibatch_index + minibatch_size * i] / sum) * tmp;
        }

        p_device_cached_exp[IDX2C(i, minibatch_index, 2 * d + 1)] = tmp;
      } else {
        p_device_alignments[minibatch_index + minibatch_size * i] = 0;
        p_device_cached_exp[IDX2C(i, minibatch_index, 2 * d + 1)] = 1; // since you divide by this
      }
    }
  }
}


/*
    Each block is responsible for multiplying one column of a h_t matrix
    alignments is laid out as:
    [minibatch] [minibatch] [minibatch] ...
*/
// CHECK: OK //
template <typename T>
__global__ void CreateCTKernel(T *p_device_alignments, T *p_device_hs_mat, T *p_device_c_t, int lstm_size, int minibatch_size, int d) {

  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < lstm_size * minibatch_size; i += gridDim.x * blockDim.x) {
    p_device_c_t[i] = 0;
    int minibatch_index = (i / lstm_size);
    for (int j = 0; j < 2 * d + 1; ++j) {
      p_device_c_t[i] += p_device_alignments[minibatch_index + minibatch_size * j] * p_device_hs_mat[i + lstm_size * minibatch_size * j];
    }
  }
}



template <typename T>
__global__ void AddTwoMatsKernel(T *p_device_mat1, T *p_device_mat2, int size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
    p_device_mat1[i] = p_device_mat1[i] + p_device_mat2[i];  
  }
}


template <typename T>
__global__ void TanhGradKernel(T *p_device_output, T *p_device_input_error, T *p_device_tanh_val, int size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
    p_device_output[i] = p_device_input_error[i] * (1 - p_device_tanh_val[i] * p_device_tanh_val[i]);
  }
}


// CHECK: OK //
template <typename T>
__global__ void TanhAttentionForwardKernel(T *p_device_output, T *p_device_in_1, T *p_device_in_2, T *p_device_bias, int lstm_size, int minibatch_size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < lstm_size * minibatch_size; i += gridDim.x * blockDim.x) {
    p_device_output[i] = ComputeTanh(p_device_in_1[i] + p_device_in_2[i] + p_device_bias[i % lstm_size]);  
  }
}


// CHECK: OK //
template <typename T>
__global__ void ZeroHT(T *p_device_h_t, int *p_deivce_01_mask, int lstm_size, int minibatch_size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < lstm_size * minibatch_size; i += gridDim.x * blockDim.x) {
    p_device_h_t[i] *= p_deivce_01_mask[i / lstm_size];  
  }
}






#define NUM_ATTENTION_THREADS 128
// used for the part 2 of the score function


// CHECK: OK //
// this is an improvement as more is done in one kernel launch
template <typename T>
__global__ void ElemReduceKernelLarge(T *p_device_h_t, T *p_device_wa_hs_tmp, T *p_device_alignments, int lstm_size, int minibatch_size, int d) {

  __shared__ T buffer[NUM_ATTENTION_THREADS];
  
  int i_start = threadIdx.x;        // start at the thread index
  int i_end = lstm_size;            // end at dim
  int i_step = blockDim.x;          // the block dimension (aka the number of threads in the block) is the step

  const int tid = threadIdx.x;

  for (int minibatch_index = blockIdx.x; minibatch_index < (2 * d + 1) * minibatch_size; minibatch_index += gridDim.x) {
    buffer[tid] = 0;

    for (int i = i_start; i < i_end; i += i_step) {
      buffer[tid] += p_device_h_t[IDX2C(i, minibatch_index, lstm_size)] * p_device_wa_hs_tmp[IDX2C(i, minibatch_index % minibatch_size, lstm_size)];
    }

    __syncthreads();

    for (int stride = NUM_ATTENTION_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        buffer[tid] += buffer[stride + tid];
      }
      __syncthreads();
    }
    __syncthreads();

    T sum_k = buffer[0];
    if (0 == tid) {
      p_device_alignments[minibatch_index] = sum_k;
    }
    __syncthreads();
  }
}



// used for the part 2 of the source function
template <typename T>
__global__ void ErrorAlignmentsKernelLarge(T *p_device_errn_to_t_ct, T *p_device_hs_mat, T *p_device_errn_to_t_as, int lstm_size, int minibatch_size, int d) {
  __shared__ T buffer[NUM_ATTENTION_THREADS];
  // start at the thread index
  int i_start = threadIdx.x;
  // end at dim
  int i_end = lstm_size;
  // the block dimension (aka the number of threads in the block) is the step
  int i_step = blockDim.x;
  const int tid = threadIdx.x;

  for (int minibatch_index = blockIdx.x; minibatch_index < (2 * d + 1) * minibatch_size; minibatch_index += gridDim.x) {
    buffer[tid] = 0;
    int s_index = minibatch_index / minibatch_size;
    for (int i = i_start; i < i_end; i += i_step) {
      // add two columns from p_device_errn_to_t_ct and p_device_hs_mat
      buffer[tid] += p_device_errn_to_t_ct[IDX2C(i, minibatch_index % minibatch_size, lstm_size)] * p_device_hs_mat[IDX2C(i, minibatch_index, lstm_size)];
    }

    __syncthreads();

    // sum
    for (int stride = NUM_ATTENTION_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        buffer[tid] += buffer[stride + tid];
      }
      __syncthreads();
    }

    __syncthreads();

    // normalize the softmax
    T sum_k = buffer[0];
    if (0 == tid) {
      p_device_errn_to_t_as[s_index + (2 * d + 1) * (minibatch_index % minibatch_size)] = sum_k;
    }

    __syncthreads();
  }
}


template <typename T>
__global__ void ErrorPTKernel(T *p_device_errn_to_t_pt, T *p_device_errn_to_t_as, int d, T sigma_sq, int *p_device_indices, int minibatch_size, T *p_device_p_t, T *p_device_alignments) {
  __shared__ T buffer[NUM_ATTENTION_THREADS];
  int minibatch_index = blockIdx.x;
  // start at the thread index
  int i_start = threadIdx.x;
  // end at dim
  int i_end = 2 * d + 1;
  // the block dimension (aka the number of threads in the block) is the step
  int i_step = blockDim.x;
  const int tid = threadIdx.x;
  buffer[tid] = 0;

  for (int i = i_start; i < i_end; i += i_step) {
    buffer[tid] += p_device_errn_to_t_as[IDX2C(i, minibatch_index, 2 * d + 1)] * p_device_alignments[IDX2C(minibatch_index, i, minibatch_size)] * ((p_device_indices[minibatch_index + i * minibatch_size] - p_device_p_t[minibatch_index]) / sigma_sq);
  }

  __syncthreads();

  for (int stride = NUM_ATTENTION_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buffer[tid] += buffer[stride + tid];
    }
    __syncthreads();
  }

  __syncthreads();

  // normalize the softmax
  T sum_k = buffer[0];
  if (0 == tid) {
    p_device_errn_to_t_pt[minibatch_index] = sum_k;
  }
}



template <typename T>
__global__ void AttentionVPErrorKernel(T *p_device_sigma, T *p_device_tanh, T *p_device_tmp_grad, T *p_device_errn_to_t_pt, int *p_device_batch_information_, int lstm_size, int minibatch_size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < lstm_size * minibatch_size; i += gridDim.x * blockDim.x) {
    int minibatch_index = i / lstm_size;
    p_device_tmp_grad[i] = p_device_errn_to_t_pt[minibatch_index] * p_device_sigma[minibatch_index] * (1 - p_device_sigma[minibatch_index]) * p_device_batch_information_[minibatch_index] * p_device_tanh[i];
  }
}


template <typename T>
__global__ void GradWPKernel(T *p_device_v_p, T *p_device_tmp, T *p_device_sigma, T *p_device_tanh, T *p_device_errn_to_t_pt, int *p_device_batch_information, int lstm_size, int minibatch_size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < lstm_size * minibatch_size; i += gridDim.x * blockDim.x) {
    int minibatch_index = i / lstm_size;
    int lstm_index = i % lstm_size;
    p_device_tmp[i] = p_device_errn_to_t_pt[minibatch_index] * p_device_batch_information[minibatch_index] * p_device_v_p[lstm_index] * p_device_sigma[minibatch_index] * (1 - p_device_sigma[minibatch_index]) * (1 - p_device_tanh[i] * p_device_tanh[i]);
  }
}


// faster w_a gradient
template <typename T>
__global__ void GetHTScalingsWaGradKernel(T *p_device_scalings, T *p_device_errn_to_t_as, T *p_device_alignments, T *p_device_cached_exp, int d, int minibatch_size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < (2 * d + 1) * minibatch_size; i += gridDim.x * blockDim.x) {
    int alignment_index = i % (2 * d + 1);
    int minibatch_index = i / (2 * d + 1);
    p_device_scalings[i] = p_device_errn_to_t_as[IDX2C(alignment_index, minibatch_index, 2 * d + 1)] * \
                           p_device_alignments[IDX2C(minibatch_index, alignment_index, minibatch_size)] * \
                           (1 - p_device_alignments[IDX2C(minibatch_index, alignment_index, minibatch_size)] / \
                           p_device_cached_exp[IDX2C(alignment_index, minibatch_index, 2 * d + 1)]);
    for (int j = 0; j < 2 * d + 1; ++j) {
      if (j != alignment_index) {
        p_device_scalings[i] += -1 * p_device_errn_to_t_as[IDX2C(j, minibatch_index, 2 * d + 1)] * \
                                p_device_alignments[IDX2C(minibatch_index, j, minibatch_size)] * \
                                p_device_alignments[IDX2C(minibatch_index, alignment_index, minibatch_size)] / \
                                p_device_cached_exp[IDX2C(alignment_index, minibatch_index, 2 * d + 1)];
      }
    }
  }
}


template <typename T>
__global__ void ScaleHTKernel(T *p_device_scalings, T *p_device_tmp_1, T *p_device_h_t, int lstm_size, int minibatch_size, int alignment_index, int d) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < lstm_size * minibatch_size; i += gridDim.x * blockDim.x) {
    int minibatch_index = i / lstm_size;
    p_device_tmp_1[i] = p_device_h_t[i] * p_device_scalings[IDX2C(alignment_index, minibatch_index, 2 * d + 1)];
  }
}


// each block will copy over one vector to the source side
template <typename T>
__global__ void CopyErrorsSource(T **p_device_total_hs_error, T *p_device_tmp_error, int *p_device_indices, int lstm_size, int minibatch_size, int d, int alignment_index, int *p_device_batch_information) {
  for (int i = blockIdx.x; i < minibatch_size; i += gridDim.x) {
    int minibatch_index = i;
    int source_index = p_device_indices[IDX2C(minibatch_index, alignment_index, minibatch_size)];
    if (-1 != source_index) {
      for (int j = threadIdx.x; j < lstm_size; j += blockDim.x) {
        //p_device_total_hs_error[source_index + p_device_batch_information[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, lstm_size)] += p_device_tmp_error[IDX2C(j, minibatch_index, lstm_size)];
        p_device_total_hs_error[p_device_batch_information[minibatch_index] - 1 - source_index + p_device_batch_information[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, lstm_size)] += p_device_tmp_error[IDX2C(j, minibatch_index, lstm_size)];
      }
    }
  }
}




// get the error for h_s from c_t
template <typename T>
__global__ void ErrorHSAndCTKernelLarge(T *p_device_errn_to_t_ct, T *p_device_alignments, int *p_device_indices, int *p_device_batch_information, T **p_device_total_hs_error, int lstm_size, int minibatch_size, int d) {
  for (int i = blockIdx.x; i < minibatch_size * (2 * d + 1); i += gridDim.x) {
    int minibatch_index = i % minibatch_size;
    int alignment_index = i / minibatch_size;
    int source_index = p_device_indices[IDX2C(minibatch_index, alignment_index, minibatch_size)];
    if (-1 != source_index) {
      for (int j = threadIdx.x; j < lstm_size; j += blockDim.x) {
        //p_device_total_hs_error[source_index + p_device_batch_information[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, lstm_size)] += p_device_errn_to_t_ct[IDX2C(j, minibatch_index, lstm_size)] * p_device_alignments[IDX2C(minibatch_index, alignment_index, minibatch_size)];  
        // src_sent = -1 -1 -1 -1 c b a
        // src_sent.length = 3
        // src_sent.no_use_length = 4
        // we want get b (index = 1), so 3 - 1 - 1 + 4 = 5, so we get b 
        p_device_total_hs_error[p_device_batch_information[minibatch_index] - 1 - source_index + p_device_batch_information[minibatch_size + minibatch_index]][IDX2C(j, minibatch_index, lstm_size)] += p_device_errn_to_t_ct[IDX2C(j, minibatch_index, lstm_size)] * p_device_alignments[IDX2C(minibatch_index, alignment_index, minibatch_size)];
      }
    }
  }
}


template <typename T>
__global__ void GradientUpdateMats(T *p_device_mat, T *p_device_mat_grad, T learning_rate, int size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
    p_device_mat[i] += learning_rate * p_device_mat_grad[i];
  }
}



////////////////////////////////////// Cell Clip //////////////////////////////////////
////////////////////////////////////// Cell Clip //////////////////////////////////////
////////////////////////////////////// Cell Clip //////////////////////////////////////
template <typename T>
__global__ void ClipMatKernel(T *p_device_mat, T threshold, int size) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
    if (p_device_mat[i] > 0) {
      p_device_mat[i] = (p_device_mat[i] > threshold) ? threshold : p_device_mat[i];  
    } else {
      p_device_mat[i] = (p_device_mat[i] < -threshold) ? -threshold : p_device_mat[i];
    }
  }
}


////////////////////////////////////// Read / Write Matrix //////////////////////////////////////
////////////////////////////////////// Read / Write Matrix //////////////////////////////////////
////////////////////////////////////// Read / Write Matrix //////////////////////////////////////
template <typename T>
void ReadMatrixGpu(T *p_device_mat, int rows, int cols, std::ifstream &input_stream) {
  T *p_tmp_mat = (T *)malloc(rows * cols * sizeof(T));

  std::string tmp_string;
  std::string tmp_token;

  for (int i = 0; i < rows; ++i) {
    std::getline(input_stream, tmp_string);
    std::istringstream iss_input(tmp_string, std::istringstream::in);
    for (int j = 0; j < cols; ++j) {
      iss_input>>tmp_token;
      p_tmp_mat[IDX2C(i, j, rows)] = std::stod(tmp_token);
    }
  }
  std::getline(input_stream, tmp_string);
  cudaMemcpy(p_device_mat, p_tmp_mat, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
  free(p_tmp_mat);
}


template <typename T>
void WriteMatrixGpu(T *p_device_mat, int rows, int cols, std::ofstream &output) {
  T *p_tmp_mat = (T *)malloc(rows * cols * sizeof(T));
  cudaMemcpy(p_tmp_mat, p_device_mat, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      output<<p_tmp_mat[IDX2C(i, j, rows)];
      if (j != cols - 1) {
        output<<" ";
      }
    }
    output<<"\n";
  }
  output<<"\n";
  free(p_tmp_mat);
}



#endif







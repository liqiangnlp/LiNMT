/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#include "deep_rnn_kernel.h"


////////////////////////////////////// atomicAdd (double) //////////////////////////////////////
////////////////////////////////////// atomicAdd (double) //////////////////////////////////////
////////////////////////////////////// atomicAdd (double) //////////////////////////////////////
//atomic add for doubles,since undefined in cuda
__device__ double atomicAddDouble(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#ifndef CUDA_8_0
//atomic add for doubles,since undefined in cuda
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif



////////////////////////////////////// Input data formatting kernels //////////////////////////////////////
////////////////////////////////////// Input data formatting kernels //////////////////////////////////////
////////////////////////////////////// Input data formatting kernels //////////////////////////////////////

// CHECK: OK //
// transform vocab indices with -1's and numbers to all 0's and 1's
__global__ void VocabTo01Kernel(int *p_device_vocab_indices_01, int *p_device_vocab_indices, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    if (-1 == p_device_vocab_indices[i]) {
      p_device_vocab_indices_01[i] = 0;
    } else {
      p_device_vocab_indices_01[i] = 1;
    }
  }
}


// CHECK: OK //
// gets rid of all -1's and replaces them with index 0
__global__ void VocabToNonMinus1Kernel(int *p_device_vocab_indices_non_minus1, int *p_device_vocab_indices, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i+= gridDim.x * blockDim.x) {
    if (-1 == p_device_vocab_indices[i]) {
      p_device_vocab_indices_non_minus1[i] = 0;
    } else {
      p_device_vocab_indices_non_minus1[i] = p_device_vocab_indices[i];
    }
  }
}



////////////////////////////////////// Forward Prop kernels //////////////////////////////////////
////////////////////////////////////// Forward Prop kernels //////////////////////////////////////
////////////////////////////////////// Forward Prop kernels //////////////////////////////////////
// Forward sigmoid kernel
__global__ void ForwardSigmoidKernel(float *p_device_final, float *p_tmp1, float *p_tmp2, float *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    float tmp_value = p_tmp1[index] + p_tmp2[index] + p_device_bias[idx];
    p_device_final[index] = 1.0f / (1.0f + expf(-1.0f * tmp_value));
  }
}

__global__ void ForwardSigmoidKernel(double *p_device_final, double *p_tmp1, double *p_tmp2, double *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    double tmp_value = p_tmp1[index] + p_tmp2[index] + p_device_bias[idx];
    p_device_final[index] = 1.0 / (1.0 + exp(-1.0 * tmp_value));
  }
  return;
}


// Forward tanh kernel
__global__ void ForwardTanhKernel(float *p_device_final, float *p_tmp1, float *p_tmp2, float *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    float tmp_value = p_tmp1[index] + p_tmp2[index] + p_device_bias[idx];
    p_device_final[index] = tanhf(tmp_value);
  }
}

__global__ void ForwardTanhKernel(double *p_device_final, double *p_tmp1, double *p_tmp2, double *p_device_bias, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    double tmp_value = p_tmp1[index] + p_tmp2[index] + p_device_bias[idx];
    p_device_final[index] = tanh(tmp_value);
  }
  return;
}


// Forward h_t kernel
__global__ void ForwardHTKernel(float *p_device_h_t, float *p_device_o_t, float *p_device_c_t, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_h_t[index] = p_device_o_t[index] * tanhf(p_device_c_t[index]);
  }
}


__global__ void ForwardHTKernel(double *p_device_h_t, double *p_device_o_t, double *p_device_c_t, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_h_t[index] = p_device_o_t[index] * tanh(p_device_c_t[index]);
  }
}




////////////////////////////////////// Backprop kernels //////////////////////////////////////
////////////////////////////////////// Backprop kernels //////////////////////////////////////
////////////////////////////////////// Backprop kernels //////////////////////////////////////

// DErrtCTKernel
__global__ void DErrtCTKernel(float *p_device_d_errt_ct, float *p_device_d_errn_to_t_ht, float *p_device_o_t, float *p_device_c_t, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    float val = tanhf(p_device_c_t[index]);
    p_device_d_errt_ct[index] = p_device_d_errn_to_t_ht[index] * p_device_o_t[index] * (1.0f - val * val);
  }
}

__global__ void DErrtCTKernel(double *p_device_d_errt_ct, double *p_device_d_errn_to_t_ht, double *p_device_o_t, double *p_device_c_t, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    double val = tanh(p_device_c_t[index]);
    p_device_d_errt_ct[index] = p_device_d_errn_to_t_ht[index] * p_device_o_t[index] * (1.0 - val * val);
  }
  return;
}


// DErrnToTOTKernel
__global__ void DErrnToTOTKernel(float *p_device_d_errn_to_t_ot, float *p_device_d_errn_to_t_ht, float *p_device_o_t, float *p_device_c_t, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_d_errn_to_t_ot[index] = p_device_d_errn_to_t_ht[index] * tanhf(p_device_c_t[index]) * p_device_o_t[index] * (1 - p_device_o_t[index]);
  }
}
__global__ void DErrnToTOTKernel(double *p_device_d_errn_to_t_ot, double *p_device_d_errn_to_t_ht, double *p_device_o_t, double *p_device_c_t, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index = IDX2C(idx, blockIdx.x, hiddenstate_size);
    p_device_d_errn_to_t_ot[index] = p_device_d_errn_to_t_ht[index] * tanh(p_device_c_t[index]) * p_device_o_t[index] * (1 - p_device_o_t[index]);
  }
  return;
}


// WGradientKernel
__global__ void WGradientKernel(float *p_device_w_grad, int *p_device_vocab_indices, float *p_tmp1, float *p_tmp2, float *p_tmp3, float *p_tmp4, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index_cols = IDX2C(idx, blockIdx.x, hiddenstate_size);
    float sum = p_tmp1[index_cols] + p_tmp2[index_cols] + p_tmp3[index_cols] + p_tmp4[index_cols];
    atomicAdd(&(p_device_w_grad[IDX2C(idx, p_device_vocab_indices[blockIdx.x], hiddenstate_size)]), sum);
  }
  return;
}

__global__ void WGradientKernel(double *p_device_w_grad, int *p_device_vocab_indices, double *p_tmp1, double *p_tmp2, double *p_tmp3, double *p_tmp4, int hiddenstate_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index_cols = IDX2C(idx, blockIdx.x, hiddenstate_size);
    double sum = p_tmp1[index_cols] + p_tmp2[index_cols] + p_tmp3[index_cols] + p_tmp4[index_cols];
    atomicAddDouble(&(p_device_w_grad[IDX2C(idx, p_device_vocab_indices[blockIdx.x], hiddenstate_size)]), sum);
  }
  return;
}


__global__ void WGradientKernelDropout(float *p_device_w_grad, int *p_device_vocab_indices, float *p_tmp1, float *p_tmp2, float *p_tmp3, float *p_tmp4, int hiddenstate_size, float *p_device_dropout_mask, float rate) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index_cols = IDX2C(idx, blockIdx.x, hiddenstate_size);
    float sum = (p_tmp1[index_cols] + p_tmp2[index_cols] + p_tmp3[index_cols] + p_tmp4[index_cols]) * (rate > p_device_dropout_mask[index_cols]) * (1 / rate);
    atomicAdd(&(p_device_w_grad[IDX2C(idx, p_device_vocab_indices[blockIdx.x], hiddenstate_size)]), sum);
  }
  return;
}

__global__ void WGradientKernelDropout(double *p_device_w_grad, int *p_device_vocab_indices, double *p_tmp1, double *p_tmp2, double *p_tmp3, double *p_tmp4, int hiddenstate_size, double *p_device_dropout_mask, double rate) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    int index_cols = IDX2C(idx, blockIdx.x, hiddenstate_size);
    double sum = (p_tmp1[index_cols] + p_tmp2[index_cols] + p_tmp3[index_cols] + p_tmp4[index_cols]) * (rate > p_device_dropout_mask[index_cols]) * (1 / rate);
    atomicAddDouble(&(p_device_w_grad[IDX2C(idx, p_device_vocab_indices[blockIdx.x], hiddenstate_size)]), sum);
  }
  return;
}



////////////////////////////////////// Softmax kernels //////////////////////////////////////
////////////////////////////////////// Softmax kernels //////////////////////////////////////
////////////////////////////////////// Softmax kernels //////////////////////////////////////

// MatrixColumnToMatrixRowKernel
__global__ void MatrixColumnToMatrixRowKernel(float *p_device_mat_final, float *p_device_mat_col, float *p_device_mat_row, int *p_device_indices, int hiddenstate_size, int output_vocab_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    atomicAdd(&p_device_mat_final[IDX2C(p_device_indices[blockIdx.x], idx, output_vocab_size)], p_device_mat_col[IDX2C(idx, blockIdx.x, hiddenstate_size)]);
  }
}


__global__ void MatrixColumnToMatrixRowKernel(double *p_device_mat_final, double *p_device_mat_col, double *p_device_mat_row, int *p_device_indices, int hiddenstate_size, int output_vocab_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < hiddenstate_size) {
    atomicAddDouble(&p_device_mat_final[IDX2C(p_device_indices[blockIdx.x], idx, output_vocab_size)], p_device_mat_col[IDX2C(idx, blockIdx.x, hiddenstate_size)]);
  }
  return;
}



// AddOnesBDGrad
__global__ void AddOnesBDGrad(float *p_device_b_d_grad, int *p_device_output_vocab_indices_01, int *p_device_output_vocab_indices, int minibatch_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < minibatch_size && 1 == p_device_output_vocab_indices_01[idx]) {
    atomicAdd(&p_device_b_d_grad[p_device_output_vocab_indices[idx]], 1);
  }
}

__global__ void AddOnesBDGrad(double *p_device_b_d_grad, int *p_device_output_vocab_indices_01, int *p_device_output_vocab_indices, int minibatch_size) {
  int idx = threadIdx.x + blockIdx.y * blockDim.x;
  if (idx < minibatch_size && 1 == p_device_output_vocab_indices_01[idx]) {
    atomicAddDouble(&p_device_b_d_grad[p_device_output_vocab_indices[idx]], 1);
  }
  return;
}







////////////////////////////////////// Attention Model //////////////////////////////////////
////////////////////////////////////// Attention Model //////////////////////////////////////
////////////////////////////////////// Attention Model //////////////////////////////////////
// CHECK: OK //
__global__ void TanhKernel(float *p_device_in, float *p_device_out, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    p_device_out[i] = tanhf(p_device_in[i]);  
  }
}


// CHECK: OK //
__global__ void TanhKernel(double *p_device_in, double *p_device_out, int total_length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total_length; i += gridDim.x * blockDim.x) {
    p_device_out[i] = tanh(p_device_in[i]);    
  }
}


// CHECK: OK //
// create indices
__global__ void CreateIndicesKernel(int *p_device_indices, int d, int minibatch_size, int *p_device_lower_upper, int *p_device_01_mask) {

  for (int i = threadIdx.x; i < minibatch_size; i += blockDim.x) {

    int current_index = p_device_lower_upper[IDX2C(0, i, 2)];
    int max_index = p_device_lower_upper[IDX2C(1, i, 2)];
    if (1 == p_device_01_mask[i]) {
      for (int j = 0; j < 2 * d + 1; ++j) {

        if (current_index > max_index) {
          p_device_indices[IDX2C(i, j, minibatch_size)] = -1;
        } else {
          p_device_indices[IDX2C(i, j, minibatch_size)] = current_index;
        }
        ++current_index;
      }
    } else {
      for (int j = 0; j < 2 * d + 1; ++j) {
        p_device_indices[IDX2C(i, j, minibatch_size)] = -1;
      }
    }
  }
}


/*
__global__
void CreateIndicesKernel(int *d_indicies, int D, int minibatch_size, int *d_lower_upper, int *d_01_mask) {

    for (int i = threadIdx.x; i < minibatch_size; i += blockDim.x) {

        int curr_index = d_lower_upper[IDX2C(0, i, 2)];
        int max_index = d_lower_upper[IDX2C(1, i, 2)];
        if (d_01_mask[i] == 1) {
            for (int j = 0; j < 2 * D + 1; j++) {

                if (curr_index > max_index) {
                    d_indicies[IDX2C(i, j, minibatch_size)] = -1;
                }
                else {
                    d_indicies[IDX2C(i, j, minibatch_size)] = curr_index;
                }
                curr_index++;
            }
        }
        else {
            for (int j = 0; j < 2 * D + 1; j++) {
                d_indicies[IDX2C(i, j, minibatch_size)] = -1;
            }
        }
    }
} */


__global__ void SetupReverseIndices(int *p_device_reverse_unique_indices, int *p_device_unique_indices, int current_num_unique) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < current_num_unique; i += gridDim.x * blockDim.x) {
    p_device_reverse_unique_indices[p_device_unique_indices[i]] = i;
  }
}





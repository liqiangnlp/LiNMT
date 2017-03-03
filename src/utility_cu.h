/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 10/28/2016
 * Time  : 11:18
 *
 */


#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <string>
#include <iostream>
#include <vector>
#include <boost/random/uniform_real.hpp>
#include <boost/random.hpp>

/* cuda */
#include "cublas_v2.h"
#include <cublas_api.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include "cuda_profiler_api.h"
#include "utility_cc.h"
#include "debug.h"

// CHECK: OK //
// This is used since all cuBLAS storage is column major
#define IDX2C(i, j, ld) (((j)*(ld))+(i))


namespace neural_machine_translation {

typedef float precision;

// add in 09/06/2016
extern bool source_side_mode__;
extern bool train_source_input_embedding_mode__;
extern bool train_target_input_embedding_mode__;
extern bool train_target_output_embedding_mode__;
extern bool train_source_rnn_mode__;
extern bool train_target_rnn_mode__;
extern bool train_attention_target_rnn_mode__;
extern bool soft_regularizer_mode__;

extern precision train_source_input_embedding_lambda__;
extern precision train_target_input_embedding_lambda__;
extern precision train_target_output_embedding_lambda__;
extern precision train_source_rnn_lambda__;
extern precision train_target_rnn_lambda__;
extern precision train_attentin_target_rnn_lambda__;



// for t-sne stuff for paper
extern precision *p_host_dump_ht__;
extern bool tsne_dump_mode__;
extern std::ofstream tsne_dump_stream__;





extern OutputLogger logger;

extern bool continue_train_mode__;
extern bool shuffle_data_mode__;

extern bool pre_normalization_mode__;

extern bool dump_every_best__;
extern int curr_dump_num__;



// stuff for unk replacement using attention
extern bool unk_replacement_mode__;
extern std::string unk_rep_file_name__;
//extern std::ofstream unk_rep_file_stream__;
extern std::vector<int> viterbi_alignments__;
extern std::vector<int> all_viterbi_alignments__;
extern std::vector<precision> alignment_scores__;       // for ensembling alignment values
extern int *p_host_align_indices__;
extern precision *p_host_alignment_values__;


extern unsigned int curr_seed__;

extern bool force_decode_mode__;

extern boost::random::mt19937 generator__;
extern double lower__;
extern double upper__;


extern bool global_grad_clip_mode__;
extern precision global_norm_clip__;
extern precision global_norm_clip_threshold__;
extern double recent_sum__;


extern bool cell_clip_mode__;
extern precision cell_clip_threshold__;
extern precision error_clip_threshold__;


extern bool individual_grad_clip_mode__;
extern precision individual_norm_clip_threshold__;

extern bool nce_score_mode__;

extern bool dump_nce_stats__;


extern bool print_partition_function_mode__;
extern std::vector<double> full_partition_values__;

void PrintPartitionStats();
}



std::string CublasErrorString(cublasStatus_t error);

void CublasErrorWrapper(cublasStatus_t cuda_stat, std::string error_message);

void CudaErrorWrapper(cudaError_t cuda_stat, std::string error_message);

void CudaGetLastError();
void CudaGetLastError(std::string message);

void CurandGenerateUniformWrapper(float *p_device_mask, int size, curandGenerator_t &generator);
void CurandGenerateUniformWrapper(double *p_device_mask, int size, curandGenerator_t &generator);

void DeviceSyncAll();


/////////////////// Inline Function ///////////////////
/////////////////// Cublas Error Wrappers ///////////////////
// CHECK: OK //
// CublasGemmWrapper
inline cublasStatus_t CublasGemmWrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *a, int lda, const float *b, int ldb, const float *beta, float *c, int ldc) {

  return cublasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

}


// CHECK: OK //
inline cublasStatus_t CublasGemmWrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *a, int lda, const double *b, int ldb, const double *beta, double *c, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* CublasGeamWrapper */
inline cublasStatus_t CublasGeamWrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *a, int lda, const float *beta, const float *b, int ldb, float *c, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c, ldc);
}


inline cublasStatus_t CublasGeamWrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *a, int lda, const double *beta, const double *b, int ldb, double *c, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c, ldc);
}


/* CublasGemvWrapper */
inline cublasStatus_t CublasGemvWrapper(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *a, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
  return cublasSgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline cublasStatus_t CublasGemvWrapper(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *a, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
  return cublasDgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}


/* CublasDgmmWrapper */
inline cublasStatus_t CublasDgmmWrapper(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float *a, int lda, const float *x, int incx, float *c, int ldc) {
  return cublasSdgmm(handle, mode, m, n, a, lda, x, incx, c, ldc);
}

inline cublasStatus_t CublasDgmmWrapper(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double *a, int lda, const double *x, int incx, double *c, int ldc) {
  return cublasDdgmm(handle, mode, m, n, a, lda, x, incx, c, ldc);
}




/////////////////// Template Function ///////////////////
template <typename T>
void InitThrustVector(thrust::host_vector<T> &h_vec, int size);


template <typename T>
void AllocateMatrixCpu(T **h_matrix, int rows, int cols);


template <typename T>
void AllocateMatrixGpu(T **d_matrix, int rows, int cols);


template <typename T>
void SetMatrixCublas(T *h_matrix, T *d_matrix, int rows, int cols);


template <typename T>
void SetVectorCublas(T *h_vector, T *d_vector, int rows);


template <typename T>
void InitializeMatrix(T *h_matrix, int rows, int cols);


template <typename T>
void InitializeMatrixOnes(T *h_matrix, int rows, int cols);


template <typename T>
void InitializeMatrixZeros(T *h_matrix, int rows, int cols);


template <typename T>
void FullMatrixSetup(T **h_matrix, T **d_matrix, int rows, int cols);


template <typename T>
void FullMatrixSetupZeros(T **h_matrix, T **d_matrix, int rows, int cols);


template <typename T>
void FullVectorSetup(T **h_vector, T **d_vector, int rows);


template <typename T>
void FullVectorSetupOnes(T **h_vector, T **d_vector, int rows);


/////////////////// Implementations for Template Function ///////////////////
// CHECK: OK //
template <typename T>
void InitThrustVector(thrust::host_vector<T> &h_vec, int size) {
  boost::uniform_real<> distributation(neural_machine_translation::lower__, neural_machine_translation::upper__);
  for (int i = 0; i < size; ++i) {
    h_vec[i] = (T)distributation(neural_machine_translation::generator__);
  }
}


// CHECK: OK //
template <typename T>
void AllocateMatrixCpu(T **h_matrix, int rows, int cols) {
  *h_matrix = (T *)malloc(rows * cols * sizeof(T));
}


// CHECK: OK //
template <typename T>
void AllocateMatrixGpu(T **d_matrix, int rows, int cols) {
  CudaErrorWrapper(cudaMalloc((void**)d_matrix, rows*cols*sizeof(T)), "GPU memory allocation failed\n");
}


// CHECK: OK //
template <typename T>
void SetMatrixCublas(T *h_matrix, T *d_matrix, int rows, int cols) {
  CublasErrorWrapper(cublasSetMatrix(rows, cols, sizeof(T), h_matrix, rows, d_matrix, rows), "cuBLAS set matrix failed\n");
}


// CHECK: OK //
template <typename T>
void SetVectorCublas(T *h_vector, T *d_vector, int rows) {
  CublasErrorWrapper(cublasSetVector(rows, sizeof(T), h_vector, 1, d_vector, 1), "cuBLAS set vector failed\n");
}


// CHECK: OK //
template <typename T>
void InitializeMatrix(T *h_matrix, int rows, int cols) {
  boost::uniform_real<> distribution(neural_machine_translation::lower__, neural_machine_translation::upper__);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      h_matrix[IDX2C(i,j,rows)] = (T)distribution(neural_machine_translation::generator__);  
    }
  }
}


// CHECK: OK //
template <typename T>
void InitializeMatrixOnes(T *h_matrix, int rows, int cols) {
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      h_matrix[IDX2C(i,j,rows)] = 1;  
    }
  }
}


// CHECK: OK //
template <typename T>
void InitializeMatrixZeros(T *h_matrix, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            h_matrix[IDX2C(i, j, rows)] = 0;
        }
    }
}


// CHECK: OK //
template <typename T>
void FullMatrixSetup(T **h_matrix, T **d_matrix, int rows, int cols) {
  AllocateMatrixCpu(h_matrix, rows, cols);
  InitializeMatrix(*h_matrix, rows, cols);
  AllocateMatrixGpu(d_matrix, rows, cols);
  SetMatrixCublas(*h_matrix, *d_matrix, rows, cols);

  free(*h_matrix);
}


// CHECK: OK //
template <typename T>
void FullMatrixSetupZeros(T **h_matrix, T **d_matrix, int rows, int cols) {
  AllocateMatrixCpu(h_matrix, rows, cols);
  InitializeMatrixZeros(*h_matrix, rows, cols);
  AllocateMatrixGpu(d_matrix, rows, cols);
  SetMatrixCublas(*h_matrix, *d_matrix, rows, cols);

  free(*h_matrix);
}


// CHECK: OK //
template <typename T>
void FullVectorSetup(T **h_vector, T **d_vector, int rows) {
  AllocateMatrixCpu(h_vector, rows, 1);
  InitializeMatrix(*h_vector, rows, 1);
  AllocateMatrixGpu(d_vector, rows, 1);
  SetVectorCublas(*h_vector, *d_vector, rows);

#ifndef CPU_DEBUG
    free(*h_vector);
#endif

}


// CHECK: OK //
template <typename T>
void FullVectorSetupOnes(T **h_vector, T **d_vector, int rows) {
  AllocateMatrixCpu(h_vector, rows, 1);
  InitializeMatrixOnes(*h_vector, rows, 1);
  AllocateMatrixGpu(d_vector, rows, 1);
  SetVectorCublas(*h_vector, *d_vector, rows);

  free(*h_vector);
}




////////////////////////////////////// cuda kernel //////////////////////////////////////
////////////////////////////////////// cuda kernel //////////////////////////////////////
////////////////////////////////////// cuda kernel //////////////////////////////////////

// CHECK: OK //
// exp(*)
__device__ inline double ComputeExp(double x) {
  return exp(x);
}

// CHECK: OK //
__device__ inline float ComputeExp(float x) {
  return expf(x);
}


/* log(*) */
__device__ inline double ComputeLog(double x) {
  return log(x);
}

__device__ inline float ComputeLog(float x) {
  return logf(x);
}


// pow(*)
// CHECK: OK //
__device__ inline double ComputePow(double x, double y) {
  return pow(x, y);
}

// CHECK: OK //
__device__ inline float ComputePow(float x, float y) {
  return powf(x, y);
}



// tanh(*)
// CHECK: OK //
__device__ inline double ComputeTanh(double x) {
  return tanh(x);
}

// CHECK: OK //
__device__ inline float ComputeTanh(float x) {
  return tanhf(x);
}



#endif



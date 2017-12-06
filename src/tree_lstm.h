/*
 *
 * Author: Qiang Li
 * Email : liqiangneu@gmail.com
 * Date  : 11/21/2017
 * Time  : 19:52
 *
 */

#ifndef TREE_LSTM_H_
#define TREE_LSTM_H_



namespace neural_machine_translation {

template<typename T>
class AnotherEncoder;

template<typename T>
class TreeLstm {

public:
  int device_number_;
  cudaStream_t s0_;
  cublasHandle_t handle_;
  int lstm_size_;
  int minibatch_size_;
  bool clip_gradients_mode_ = false;
  T norm_clip_;

public:
  AnotherEncoder<T> *p_another_encoder_;

public:
  T *p_device_ones_minibatch_;

  T *p_device_child_ht_1_;
  T *p_device_child_ht_2_;
  T *p_device_child_ct_1_;
  T *p_device_child_ct_2_;

public:
  T *p_device_i_t_;
  T *p_device_f_t_1_;
  T *p_device_f_t_2_;
  T *p_device_c_prime_t_tanh_;
  T *p_device_o_t_;
  T *p_device_c_t_;
  T *p_device_h_t_;

public:
  // parameters
  // biases
  T *p_device_b_i_;
  T *p_device_b_f_;   // initialize to one
  T *p_device_b_o_;
  T *p_device_b_c_;

  T *p_device_m_i_1_;
  T *p_device_m_f_1_;
  T *p_device_m_o_1_;
  T *p_device_m_c_1_;
  T *p_device_m_i_2_;
  T *p_device_m_f_2_;
  T *p_device_m_o_2_;
  T *p_device_m_c_2_;

public:
  // tmp stuff
  T *p_device_tmp_1_;
  T *p_device_tmp_2_;
  T *p_device_tmp_3_;
  T *p_device_tmp_4_;
  T *p_device_tmp_5_;
  T *p_device_tmp_6_;
  T *p_device_tmp_7_;
  T *p_device_tmp_8_;

public:
  TreeLstm(int lstm_size,int device_number, AnotherEncoder<T> *p_another_encoder);
  void LoadWeights(std::ifstream &input_stream);

public:
  void Forward();
};



template<typename T>
TreeLstm<T>::TreeLstm(int lstm_size, int device_number, AnotherEncoder<T> *p_another_encoder) {
  device_number_ = device_number;
  minibatch_size_ = 1;
  lstm_size_ = lstm_size;
  p_another_encoder_ = p_another_encoder;

  cudaSetDevice(device_number);

  CublasErrorWrapper(cublasCreate(&handle_), "CUBLAS handler initialization failed\n");
  cudaStreamCreate(&s0_);

  T *p_host_tmp;
  FullVectorSetupOnes(&p_host_tmp, &p_device_ones_minibatch_, minibatch_size_);

  FullMatrixSetup(&p_host_tmp, &p_device_child_ht_1_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_child_ht_2_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_child_ct_1_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_child_ct_2_, lstm_size, minibatch_size_);

  FullMatrixSetup(&p_host_tmp, &p_device_i_t_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_f_t_1_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_f_t_2_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_c_prime_t_tanh_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_o_t_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_c_t_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_h_t_, lstm_size, minibatch_size_);

  FullMatrixSetup(&p_host_tmp, &p_device_b_i_, lstm_size, 1);
  FullMatrixSetup(&p_host_tmp, &p_device_b_f_, lstm_size, 1);
  FullMatrixSetup(&p_host_tmp, &p_device_b_o_, lstm_size, 1);
  FullMatrixSetup(&p_host_tmp, &p_device_b_c_, lstm_size, 1);

  FullMatrixSetup(&p_host_tmp, &p_device_m_i_1_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_f_1_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_o_1_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_c_1_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_i_2_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_f_2_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_o_2_, lstm_size, lstm_size);
  FullMatrixSetup(&p_host_tmp, &p_device_m_c_2_, lstm_size, lstm_size);

  FullMatrixSetup(&p_host_tmp, &p_device_tmp_1_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_2_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_3_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_4_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_5_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_6_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_7_, lstm_size, minibatch_size_);
  FullMatrixSetup(&p_host_tmp, &p_device_tmp_8_, lstm_size, minibatch_size_);
}


template<typename T>
void TreeLstm<T>::LoadWeights(std::ifstream &input_stream) {
  cudaSetDevice(device_number_);

  ReadMatrixGpu(p_device_m_i_1_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_f_1_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_o_1_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_c_1_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_i_2_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_f_2_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_o_2_, lstm_size_, lstm_size_, input_stream);
  ReadMatrixGpu(p_device_m_c_2_, lstm_size_, lstm_size_, input_stream);

  ReadMatrixGpu(p_device_b_i_, lstm_size_, 1, input_stream);
  ReadMatrixGpu(p_device_b_f_, lstm_size_, 1, input_stream);
  ReadMatrixGpu(p_device_b_o_, lstm_size_, 1, input_stream);
  ReadMatrixGpu(p_device_b_c_, lstm_size_, 1, input_stream);
}


template<typename T>
void TreeLstm<T>::Forward() {
  cudaSetDevice(device_number_);

  T alpha = 1;
  T beta = 0;

  int threads_per_block = 128;
  int num_block = (lstm_size_ + threads_per_block - 1) / threads_per_block;
  dim3 kernel(minibatch_size_, num_block, 1);

  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_i_1_, lstm_size_, \
                     p_device_child_ht_1_, lstm_size_, &beta, p_device_tmp_1_, lstm_size_), "Forward prop i_t tmp1 failed\n");

  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_i_2_, lstm_size_, \
                     p_device_child_ht_2_, lstm_size_, &beta, p_device_tmp_2_, lstm_size_), "Forward prop i_t tmp2 failed\n");

  ForwardSigmoidKernel<<<kernel, threads_per_block, 0, s0_>>>(p_device_i_t_, p_device_tmp_1_, p_device_tmp_2_, p_device_b_i_, lstm_size_);
  CudaGetLastError("i_t tree lstm");

  alpha = 1;
  beta = 0;

  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_f_1_, lstm_size_, \
                     p_device_child_ht_1_, lstm_size_, &beta, p_device_tmp_3_, lstm_size_), "Forward prop f_t tmp3 failed\n");
  ForwardSigmoidKernelSmall<<<kernel, threads_per_block, 0, s0_>>>(p_device_f_t_1_, p_device_tmp_3_, p_device_b_f_, lstm_size_);

  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_f_2_, lstm_size_, \
                     p_device_child_ht_2_, lstm_size_, &beta, p_device_tmp_4_, lstm_size_), "Forward prop f_t tmp4 failed\n");
  ForwardSigmoidKernelSmall<<<kernel, threads_per_block, 0, s0_>>>(p_device_f_t_2_, p_device_tmp_4_, p_device_b_f_, lstm_size_);

  alpha = 1;
  beta = 0;
  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_c_1_, lstm_size_, \
                     p_device_child_ht_1_, lstm_size_, &beta, p_device_tmp_5_, lstm_size_), "Forward prop c_prime_t_tanh tmp5 failed\n");
  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_c_2_, lstm_size_, \
                     p_device_child_ht_2_, lstm_size_, &beta, p_device_tmp_6_, lstm_size_), "Forward prop c_prime_t_tanh tmp6 failed\n");
  ForwardTanhKernel<<<kernel, threads_per_block, 0, s0_>>>(p_device_c_prime_t_tanh_, p_device_tmp_5_, p_device_tmp_6_, p_device_b_c_, lstm_size_);
  CudaGetLastError("c_prime_t_tanh");

  alpha = 1;
  beta = 0;
  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_o_1_, lstm_size_, \
                     p_device_child_ht_1_, lstm_size_, &beta, p_device_tmp_7_, lstm_size_), "Forward prop o_t tmp1 failed\n");
  cublasSetStream(handle_, s0_);
  CublasErrorWrapper(CublasGemmWrapper(handle_, CUBLAS_OP_N, CUBLAS_OP_N, lstm_size_, minibatch_size_, lstm_size_, &alpha, p_device_m_o_2_, lstm_size_, \
                     p_device_child_ht_2_, lstm_size_, &beta, p_device_tmp_8_, lstm_size_), "Forward prop o_t tmp2 failed\n");
  ForwardSigmoidKernel<<<kernel, threads_per_block, 0, s0_>>>(p_device_o_t_, p_device_tmp_7_, p_device_tmp_8_, p_device_b_o_, lstm_size_);
  CudaGetLastError("o_t");

  ForwardCTKernelTree<<<kernel, threads_per_block, 0, s0_>>>(p_device_c_t_, p_device_f_t_1_, p_device_child_ct_1_, p_device_f_t_2_, p_device_child_ct_2_, p_device_i_t_, p_device_c_prime_t_tanh_, lstm_size_);
  CudaGetLastError("c_t");

  if (cell_clip_mode__) {
    ;
  }

  ForwardHTKernel<<<kernel, threads_per_block, 0, s0_>>>(p_device_h_t_, p_device_o_t_, p_device_c_t_, lstm_size_);
  CudaGetLastError("h_t");
}


} // end of neural_machine_translation namespace



#endif


